"""Immutable time splits with a label-blind, isolated final holdout.

This module prepares keys and diagnostics only. It never reads target values,
fits a model, or scores the final holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = ["pkg_no", "pay_time", "issue_time"]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{field} must be a nonempty safe identifier")
    return value


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None:
        raise ValueError(f"{field} must be a valid naive timestamp in the original data time convention")
    return result


@dataclass(frozen=True)
class FoldSpec:
    name: str
    F: pd.Timestamp
    E: pd.Timestamp
    C: pd.Timestamp
    H_days: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any], name: str | None = None) -> "FoldSpec":
        fold_name = _identifier(name if name is not None else value.get("name"), "fold name")
        horizon = value.get("H_days")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"{fold_name}.H_days must be a positive integer")
        result = cls(
            name=fold_name,
            F=_timestamp(value.get("F"), f"{fold_name}.F"),
            E=_timestamp(value.get("E"), f"{fold_name}.E"),
            C=_timestamp(value.get("C"), f"{fold_name}.C"),
            H_days=horizon,
        )
        if not result.F < result.E <= result.C:
            raise ValueError(f"{fold_name} requires F < E <= C")
        return result

    @property
    def end(self) -> pd.Timestamp:
        return self.C + pd.Timedelta(days=self.H_days)

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "F": self.F.isoformat(), "E": self.E.isoformat(),
                "C": self.C.isoformat(), "H_days": self.H_days, "eval_end": self.end.isoformat()}


def validate_config(config: dict[str, Any]) -> tuple[str, FoldSpec, list[FoldSpec]]:
    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a mapping")
    split_id = _identifier(config.get("split_id"), "split_id")
    if not isinstance(config.get("holdout"), dict):
        raise ValueError("holdout specification is required")
    holdout = FoldSpec.from_mapping(config["holdout"], name="holdout")
    values = config.get("development")
    if not isinstance(values, list) or not values:
        raise ValueError("At least one development specification is required")
    development = [FoldSpec.from_mapping(value) for value in values]
    names = [fold.name for fold in development]
    if "holdout" in names or len(set(names)) != len(names):
        raise ValueError("Development names must be unique and cannot be holdout")
    return split_id, holdout, development


def normalize_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split features: {sorted(missing)}")
    result = frame[FEATURE_COLUMNS].copy()
    result["pkg_no"] = result["pkg_no"].astype("string")
    if result["pkg_no"].isna().any() or result["pkg_no"].str.len().eq(0).any():
        raise ValueError("pkg_no cannot be null or empty")
    if result["pkg_no"].duplicated().any():
        raise ValueError("pkg_no must be unique before splitting")
    for column in ["pay_time", "issue_time"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"Invalid {column}; resolve time anomalies before freezing splits")
        if result[column].dt.tz is not None:
            raise ValueError(f"{column} must retain the original naive time convention")
    return result


def holdout_keys(features: pd.DataFrame, spec: FoldSpec) -> pd.DataFrame:
    """Uses only decision-time features, never sign_time or y_total_hours."""
    mask = features["issue_time"].ge(spec.C) & features["issue_time"].lt(spec.end)
    keys = features.loc[mask, ["pkg_no"]].reset_index(drop=True)
    if keys.empty:
        raise ValueError("Final holdout is empty")
    return keys


def _key_digest(keys: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in sorted(keys["pkg_no"].astype(str).tolist()):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _write_keys_once(path: Path, keys: pd.DataFrame) -> dict[str, Any]:
    expected = _key_digest(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path, columns=["pkg_no"])
        if len(existing) != len(keys) or existing["pkg_no"].duplicated().any() or _key_digest(existing) != expected:
            raise ValueError(f"Refusing to overwrite frozen keys: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        keys[["pkg_no"]].to_parquet(temporary, index=False)
        temporary.replace(path)
    return {"path": str(path), "n": len(keys), "keys_sha256": expected, "file_sha256": _sha256_file(path)}


def _bounds(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n": len(frame),
        **{f"{column}_{edge}": getattr(frame[column], edge)().isoformat() if len(frame) else None
           for column in ["pay_time", "issue_time"] for edge in ["min", "max"]},
    }


def _carrier_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{"lgs_company_nm": None if pd.isna(carrier) else str(carrier), "n": int(count)}
            for carrier, count in frame.groupby("lgs_company_nm", dropna=False).size().items()]


def _maturity(queue: pd.DataFrame, cutoff: pd.Timestamp) -> dict[str, Any]:
    records = []
    for carrier, group in queue.groupby("lgs_company_nm", dropna=False):
        mature = int(group["sign_time"].lt(cutoff).sum())
        records.append({"lgs_company_nm": None if pd.isna(carrier) else str(carrier),
                        "queue_n": len(group), "mature_n": mature,
                        "excluded_n": len(group) - mature, "mature_fraction": mature / len(group)})
    mature = int(queue["sign_time"].lt(cutoff).sum())
    return {"denominator": "official signed-training rows in the predeclared calibration issue queue",
            "queue_n": len(queue), "mature_n": mature, "excluded_n": len(queue) - mature,
            "mature_fraction": mature / len(queue) if len(queue) else None,
            "by_carrier": records}


def _checked_feature_masks(features: pd.DataFrame, spec: FoldSpec, frozen_holdout: pd.DataFrame,
                           final_holdout: bool) -> tuple[dict[str, pd.Series], pd.Series]:
    issue = features["issue_time"]
    isolated = features["pkg_no"].isin(frozen_holdout["pkg_no"])
    masks = {"fit": issue.lt(spec.F), "cal": issue.ge(spec.F) & issue.lt(spec.E),
             "eval": issue.ge(spec.C) & issue.lt(spec.end)}
    for phase, mask in masks.items():
        if not (final_holdout and phase == "eval") and (mask & isolated).any():
            raise ValueError(f"{spec.name}.{phase} overlaps final holdout; choose and freeze nonconflicting windows")
    if final_holdout and not masks["eval"].equals(isolated):
        raise ValueError("Final evaluation keys differ from the frozen holdout")
    return masks, isolated


def build_fold(features: pd.DataFrame, label_metadata: pd.DataFrame, spec: FoldSpec,
               frozen_holdout: pd.DataFrame, *, final_holdout: bool = False) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Construct keys; final-holdout labels are removed before any label parsing."""
    feature_masks, isolated = _checked_feature_masks(features, spec, frozen_holdout, final_holdout)
    needed = {"pkg_no", "sign_time", "lgs_company_nm"}
    if needed - set(label_metadata):
        raise ValueError(f"Missing label metadata columns: {sorted(needed - set(label_metadata))}")
    metadata = label_metadata[list(needed)].copy()
    metadata["pkg_no"] = metadata["pkg_no"].astype("string")
    if metadata["pkg_no"].duplicated().any() or len(metadata) != len(features):
        raise ValueError("Label metadata must have exactly one row per feature key")
    # Erase isolated signs before conversion/validity checks. Even invalid or future
    # rewritten holdout signs cannot change development masks or diagnostics.
    metadata.loc[metadata["pkg_no"].isin(frozen_holdout["pkg_no"]), "sign_time"] = pd.NaT
    metadata["sign_time"] = pd.to_datetime(metadata["sign_time"], errors="coerce")
    joined = features.merge(metadata, on="pkg_no", how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("Feature and label metadata key sets differ")
    if joined.loc[~isolated.to_numpy(), "sign_time"].isna().any():
        raise ValueError("Invalid sign_time outside final holdout; resolve before making splits")
    # merge preserves left row order; masks are reset to avoid relying on index labels.
    feature_masks = {phase: mask.to_numpy() for phase, mask in feature_masks.items()}
    masks = {"fit": feature_masks["fit"] & joined["sign_time"].lt(spec.F).to_numpy(),
             "cal": feature_masks["cal"] & joined["sign_time"].lt(spec.C).to_numpy(),
             "eval": feature_masks["eval"]}
    keys = {phase: joined.loc[mask, ["pkg_no"]].reset_index(drop=True) for phase, mask in masks.items()}
    if any(value.empty for value in keys.values()):
        raise ValueError(f"{spec.name} has an empty fit/cal/eval segment")
    phase_names = list(keys)
    for index, phase in enumerate(phase_names):
        for other in phase_names[index + 1:]:
            if keys[phase]["pkg_no"].isin(keys[other]["pkg_no"]).any():
                raise ValueError(f"{spec.name}: {phase}/{other} keys overlap")
    diagnostics: dict[str, Any] = {"protocol": "asof_backtest", "times": spec.metadata(), "segments": {}}
    for phase, mask in masks.items():
        selected = joined.loc[mask]
        detail = {**_bounds(selected), "carriers": _carrier_counts(selected)}
        if phase != "eval":
            detail["max_used_sign_time"] = selected["sign_time"].max().isoformat()
        diagnostics["segments"][phase] = detail
    diagnostics["calibration_maturity"] = _maturity(joined.loc[feature_masks["cal"]], spec.C)
    diagnostics["history_source"] = "fit_ids.parquet only; per-target historical time guards still required"
    diagnostics["eval_labels_used"] = False
    return keys, diagnostics


def make_splits(data_dir: Path, config_path: Path, out_base: Path,
                report_path: Path | None = None) -> dict[str, Any]:
    data_dir, config_path, out_base = map(Path, (data_dir, config_path, out_base))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_id, holdout_spec, development = validate_config(config)
    canonical = {"split_id": split_id, "holdout": holdout_spec.metadata(),
                 "development": [fold.metadata() for fold in development]}
    config_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    train_path = data_dir / "train.parquet"
    data_sha = _sha256_file(train_path)
    report_path = Path(report_path) if report_path is not None else PROJECT_ROOT / "reports/splits" / f"{split_id}.json"
    split_root = out_base / split_id
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != config_sha or previous.get("train_parquet_sha256") != data_sha:
            raise ValueError("Frozen split provenance changed; use a new split_id, do not overwrite")
    # Phase 1: holdout freezing is strictly feature-only.
    features = normalize_features(pd.read_parquet(train_path, columns=FEATURE_COLUMNS))
    frozen = holdout_keys(features, holdout_spec)
    for spec in development:
        _checked_feature_masks(features, spec, frozen, final_holdout=False)
    holdout_file = _write_keys_once(split_root / "holdout_ids.parquet", frozen)
    holdout_rows = features.loc[features["pkg_no"].isin(frozen["pkg_no"])]
    report = {"split_id": split_id, "status": "holdout_locked", "config": canonical,
              "config_sha256": config_sha, "config_file_sha256": _sha256_file(config_path),
              "train_parquet_sha256": data_sha, "train_parquet": str(train_path),
              "holdout": {**holdout_file, **_bounds(holdout_rows)}, "folds": {}}
    _atomic_json(report_path, report)
    # Phase 2: only after the immutable key list/report exists may maturity labels
    # be read. No y_total_hours is ever requested by this module.
    metadata = pd.read_parquet(train_path, columns=["pkg_no", "sign_time", "lgs_company_nm"])
    for spec in [*development, holdout_spec]:
        phase_keys, diagnostics = build_fold(features, metadata, spec, frozen,
                                             final_holdout=spec.name == "holdout")
        diagnostics["artifacts"] = {phase: _write_keys_once(split_root / spec.name / f"{phase}_ids.parquet", ids)
                                    for phase, ids in phase_keys.items()}
        report["folds"][spec.name] = diagnostics
        _atomic_json(report_path, report)
    report["status"] = "complete"
    report["holdout_isolation_checked"] = True
    _atomic_json(report_path, report)
    return report

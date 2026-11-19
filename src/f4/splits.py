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
import shutil
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_COLUMNS = ["pkg_no", "pay_time", "issue_time"]
LEGACY_PROTOCOL = "asof_backtest"
DEPLOY_C_PROTOCOL = "asof_deploy_C_v1"
PROTOCOLS = {LEGACY_PROTOCOL, DEPLOY_C_PROTOCOL}


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
    protocol: str = LEGACY_PROTOCOL
    model_fit_time: pd.Timestamp | None = None
    fit_label_cutoff: pd.Timestamp | None = None
    cal_label_cutoff: pd.Timestamp | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], name: str | None = None,
                     protocol: str = LEGACY_PROTOCOL) -> "FoldSpec":
        if protocol not in PROTOCOLS:
            raise ValueError(f"Unknown split protocol: {protocol}")
        if value.get("protocol", protocol) != protocol:
            raise ValueError("Fold protocol disagrees with split protocol")
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
            protocol=protocol,
        )
        if not result.F < result.E <= result.C:
            raise ValueError(f"{fold_name} requires F < E <= C")
        expected = {"model_fit_time": result.F if protocol == LEGACY_PROTOCOL else result.C,
                    "fit_label_cutoff": result.F if protocol == LEGACY_PROTOCOL else result.C,
                    "cal_label_cutoff": result.C}
        resolved = {}
        for field, cutoff in expected.items():
            if protocol == DEPLOY_C_PROTOCOL and field not in value:
                raise ValueError(f"{fold_name}.{field} must be explicit for {DEPLOY_C_PROTOCOL}")
            supplied = _timestamp(value[field], f"{fold_name}.{field}") if field in value else cutoff
            if supplied != cutoff:
                raise ValueError(f"{fold_name}.{field} disagrees with protocol {protocol}")
            resolved[field] = supplied
        return cls(result.name, result.F, result.E, result.C, result.H_days,
                   protocol, **resolved)

    @property
    def end(self) -> pd.Timestamp:
        return self.C + pd.Timedelta(days=self.H_days)

    def metadata(self) -> dict[str, Any]:
        result = {"name": self.name, "F": self.F.isoformat(), "E": self.E.isoformat(),
                  "C": self.C.isoformat(), "H_days": self.H_days, "eval_end": self.end.isoformat()}
        # The legacy canonical shape is deliberately unchanged: old split hashes
        # and frozen reports remain valid after adding this protocol.
        if self.protocol != LEGACY_PROTOCOL:
            result.update(protocol=self.protocol,
                          model_fit_time=self.model_fit_time.isoformat(),
                          fit_label_cutoff=self.fit_label_cutoff.isoformat(),
                          cal_label_cutoff=self.cal_label_cutoff.isoformat())
        return result

    def cutoff(self, phase: str) -> pd.Timestamp:
        if phase == "fit":
            return self.fit_label_cutoff if self.fit_label_cutoff is not None else self.F
        if phase == "cal":
            return self.cal_label_cutoff if self.cal_label_cutoff is not None else self.C
        raise ValueError(f"No label cutoff for phase: {phase}")


def validate_config(config: dict[str, Any]) -> tuple[str, FoldSpec, list[FoldSpec]]:
    if not isinstance(config, dict):
        raise ValueError("Split configuration must be a mapping")
    split_id = _identifier(config.get("split_id"), "split_id")
    protocol = config.get("protocol", LEGACY_PROTOCOL)
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown split protocol: {protocol}")
    if not isinstance(config.get("holdout"), dict):
        raise ValueError("holdout specification is required")
    holdout = FoldSpec.from_mapping(config["holdout"], name="holdout", protocol=protocol)
    values = config.get("development")
    if not isinstance(values, list) or not values:
        raise ValueError("At least one development specification is required")
    development = [FoldSpec.from_mapping(value, protocol=protocol) for value in values]
    names = [fold.name for fold in development]
    if "holdout" in names or len(set(names)) != len(names):
        raise ValueError("Development names must be unique and cannot be holdout")
    if protocol == DEPLOY_C_PROTOCOL:
        reference = config.get("reference")
        required = {"split_id", "report", "report_sha256", "artifacts"}
        if not isinstance(reference, dict) or required - set(reference):
            raise ValueError("Deployment-C split requires a pinned reference split/report/artifacts")
        reference_id = _identifier(reference["split_id"], "reference split_id")
        if reference_id == split_id:
            raise ValueError("Deployment-C must use a new split_id")
        if not isinstance(reference["report_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", reference["report_sha256"]):
            raise ValueError("reference.report_sha256 must be a SHA256 digest")
        for field in ["report", "artifacts"]:
            if not isinstance(reference[field], str) or not reference[field].strip():
                raise ValueError(f"reference.{field} must be a path")
    return split_id, holdout, development


def canonical_split_config(config: dict[str, Any]) -> dict[str, Any]:
    split_id, holdout, development = validate_config(config)
    result = {"split_id": split_id, "holdout": holdout.metadata(),
              "development": [fold.metadata() for fold in development]}
    if holdout.protocol != LEGACY_PROTOCOL:
        result.update(protocol=holdout.protocol, reference=dict(config["reference"]))
    return result


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


def _project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _reference_keys(reference: dict, phase: str, fold: str | None = None) -> tuple[Path, pd.DataFrame, dict]:
    report = reference["report"]
    record = report["holdout"] if fold is None else report["folds"][fold]["artifacts"][phase]
    source = reference["root"] / ("holdout_ids.parquet" if fold is None else f"{fold}/{phase}_ids.parquet")
    if _sha256_file(source) != record["file_sha256"]:
        raise ValueError(f"Reference frozen key file changed: {source}")
    keys = pd.read_parquet(source, columns=["pkg_no"], use_threads=False)
    if (keys["pkg_no"].isna().any() or keys["pkg_no"].duplicated().any()
            or len(keys) != record["n"] or _key_digest(keys) != record["keys_sha256"]):
        raise ValueError(f"Reference frozen key contents changed: {source}")
    return source, keys, record


def _copy_reference_keys(reference: dict, target: Path, phase: str,
                         fold: str | None = None) -> dict:
    source, keys, recorded = _reference_keys(reference, phase, fold)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256_file(target) != recorded["file_sha256"]:
            raise ValueError(f"Copied reference key file changed: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    if _sha256_file(target) != recorded["file_sha256"]:
        raise ValueError(f"Reference key copy failed: {target}")
    return {"path": str(target), "n": len(keys), "keys_sha256": recorded["keys_sha256"],
            "file_sha256": recorded["file_sha256"]}


def _load_reference(config: dict, data_sha: str, holdout: FoldSpec,
                    development: list[FoldSpec]) -> dict | None:
    if holdout.protocol == LEGACY_PROTOCOL:
        return None
    declared = config["reference"]
    path = _project_path(declared["report"])
    if _sha256_file(path) != declared["report_sha256"]:
        raise ValueError("Pinned reference report changed")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (report.get("status") != "complete" or report.get("holdout_isolation_checked") is not True
            or report.get("split_id") != declared["split_id"]
            or report.get("train_parquet_sha256") != data_sha
            or report.get("config_sha256") != hashlib.sha256(
                json.dumps(report["config"], sort_keys=True).encode()).hexdigest()):
        raise ValueError("Reference split/source provenance does not match")
    if declared.get("config") is not None and _sha256_file(_project_path(declared["config"])) != report["config_file_sha256"]:
        raise ValueError("Reference split configuration changed")
    old_specs = {item["name"]: item for item in report["config"]["development"]}
    if set(old_specs) != {spec.name for spec in development}:
        raise ValueError("Deployment-C development folds must match the reference")
    fields = ["name", "F", "E", "C", "H_days", "eval_end"]
    for spec in [*development, holdout]:
        old = report["config"]["holdout"] if spec.name == "holdout" else old_specs[spec.name]
        if {field: spec.metadata()[field] for field in fields} != old:
            raise ValueError("Deployment-C must preserve reference issue-time windows")
        if report["folds"][spec.name].get("protocol", LEGACY_PROTOCOL) != LEGACY_PROTOCOL:
            raise ValueError("Deployment-C reference must be a legacy frozen split")
    reference = {"report": report, "root": _project_path(declared["artifacts"]),
                 "path": path, "sha256": declared["report_sha256"]}
    # Check every pinned artifact, including fit, before any source signature
    # metadata is requested. A valid report hash alone does not freeze its files.
    _reference_keys(reference, "eval")
    for spec in [*development, holdout]:
        for phase in ["fit", "cal", "eval"]:
            _reference_keys(reference, phase, spec.name)
    return reference


def verify_split_reference(config: dict, report: dict, data_sha: str,
                           holdout: FoldSpec, development: list[FoldSpec],
                           split_root: Path) -> None:
    """Recheck C's pinned source and inherited files before reading labels.

    Legacy splits have no reference and retain their original validation path.
    This check deliberately covers all inherited windows, even when a caller
    prepares only one development fold.
    """
    reference = _load_reference(config, data_sha, holdout, development)
    if reference is None:
        return
    expected = {"split_id": reference["report"]["split_id"],
                "report": str(reference["path"]), "report_sha256": reference["sha256"],
                "unchanged_phases": ["cal", "eval", "holdout"], "old_fit_is_subset": True}
    if report.get("reference") != expected:
        raise ValueError("Deployment-C reference provenance does not match")
    split_root = Path(split_root)
    copied = [(split_root / "holdout_ids.parquet", report["holdout"], reference["report"]["holdout"])]
    for spec in [*development, holdout]:
        for phase in ["cal", "eval"]:
            copied.append((split_root / spec.name / f"{phase}_ids.parquet",
                           report["folds"][spec.name]["artifacts"][phase],
                           reference["report"]["folds"][spec.name]["artifacts"][phase]))
    fields = ["n", "keys_sha256", "file_sha256"]
    for path, current, original in copied:
        if ({field: current[field] for field in fields} != {field: original[field] for field in fields}
                or _sha256_file(path) != original["file_sha256"]):
            raise ValueError(f"Inherited reference key file changed: {path}")


def _label_metadata_without_holdout(train_path: Path, features: pd.DataFrame,
                                    frozen: pd.DataFrame) -> pd.DataFrame:
    """Read signature metadata only for allowed keys; retain shared carriers for holdout."""
    isolated = features["pkg_no"].isin(frozen["pkg_no"])
    allowed = features.loc[~isolated, "pkg_no"].tolist()
    metadata = pd.read_parquet(train_path, columns=["pkg_no", "sign_time", "lgs_company_nm"],
                               filters=[("pkg_no", "in", allowed)], use_threads=False)
    if (metadata["pkg_no"].isna().any() or metadata["pkg_no"].duplicated().any()
            or len(metadata) != len(allowed) or not metadata["pkg_no"].isin(allowed).all()):
        raise ValueError("Filtered signature metadata does not match non-holdout keys")
    carriers = pd.read_parquet(train_path, columns=["pkg_no", "lgs_company_nm"],
                               filters=[("pkg_no", "in", frozen["pkg_no"].tolist())], use_threads=False)
    if (carriers["pkg_no"].duplicated().any() or len(carriers) != len(frozen)
            or _key_digest(carriers[["pkg_no"]]) != _key_digest(frozen)):
        raise ValueError("Holdout shared carrier metadata does not match frozen keys")
    carriers["sign_time"] = pd.NaT
    return pd.concat([metadata, carriers], ignore_index=True)


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
    masks = {"fit": feature_masks["fit"] & joined["sign_time"].lt(spec.cutoff("fit")).to_numpy(),
             "cal": feature_masks["cal"] & joined["sign_time"].lt(spec.cutoff("cal")).to_numpy(),
             "eval": feature_masks["eval"]}
    keys = {phase: joined.loc[mask, ["pkg_no"]].reset_index(drop=True) for phase, mask in masks.items()}
    if any(value.empty for value in keys.values()):
        raise ValueError(f"{spec.name} has an empty fit/cal/eval segment")
    phase_names = list(keys)
    for index, phase in enumerate(phase_names):
        for other in phase_names[index + 1:]:
            if keys[phase]["pkg_no"].isin(keys[other]["pkg_no"]).any():
                raise ValueError(f"{spec.name}: {phase}/{other} keys overlap")
    diagnostics: dict[str, Any] = {"protocol": spec.protocol, "times": spec.metadata(), "segments": {}}
    for phase, mask in masks.items():
        selected = joined.loc[mask]
        detail = {**_bounds(selected), "carriers": _carrier_counts(selected)}
        if phase != "eval":
            detail["max_used_sign_time"] = selected["sign_time"].max().isoformat()
        diagnostics["segments"][phase] = detail
    diagnostics["calibration_maturity"] = _maturity(joined.loc[feature_masks["cal"]], spec.cutoff("cal"))
    diagnostics["history_source"] = "fit_ids.parquet only; per-target historical time guards still required"
    diagnostics["eval_labels_used"] = False
    return keys, diagnostics


def make_splits(data_dir: Path, config_path: Path, out_base: Path,
                report_path: Path | None = None) -> dict[str, Any]:
    data_dir, config_path, out_base = map(Path, (data_dir, config_path, out_base))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_id, holdout_spec, development = validate_config(config)
    canonical = canonical_split_config(config)
    config_sha = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    train_path = data_dir / "train.parquet"
    data_sha = _sha256_file(train_path)
    reference = _load_reference(config, data_sha, holdout_spec, development)
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
    if reference is not None:
        _, reference_holdout, _ = _reference_keys(reference, "eval")
        if _key_digest(frozen) != _key_digest(reference_holdout):
            raise ValueError("Deployment-C holdout keys differ from reference")
        holdout_file = _copy_reference_keys(reference, split_root / "holdout_ids.parquet", "eval")
        for spec in [*development, holdout_spec]:
            issue_eval = features.loc[features["issue_time"].ge(spec.C)
                                       & features["issue_time"].lt(spec.end), ["pkg_no"]]
            _, old_eval, _ = _reference_keys(reference, "eval", spec.name)
            if _key_digest(issue_eval) != _key_digest(old_eval):
                raise ValueError("Deployment-C evaluation keys differ from reference")
            for phase in ["cal", "eval"]:
                _copy_reference_keys(reference, split_root / spec.name / f"{phase}_ids.parquet", phase, spec.name)
    else:
        holdout_file = _write_keys_once(split_root / "holdout_ids.parquet", frozen)
    holdout_rows = features.loc[features["pkg_no"].isin(frozen["pkg_no"])]
    report = {"split_id": split_id, "status": "holdout_locked", "config": canonical,
              "config_sha256": config_sha, "config_file_sha256": _sha256_file(config_path),
              "train_parquet_sha256": data_sha, "train_parquet": str(train_path),
              "holdout": {**holdout_file, **_bounds(holdout_rows)}, "folds": {}}
    if reference is not None:
        report["protocol"] = holdout_spec.protocol
        report["reference"] = {"split_id": reference["report"]["split_id"],
                               "report": str(reference["path"]), "report_sha256": reference["sha256"],
                               "unchanged_phases": ["cal", "eval", "holdout"],
                               "old_fit_is_subset": False}
    _atomic_json(report_path, report)
    # Phase 2: only after the immutable key list/report exists may maturity labels
    # be read. No y_total_hours is ever requested by this module.
    metadata = _label_metadata_without_holdout(train_path, features, frozen)
    for spec in [*development, holdout_spec]:
        phase_keys, diagnostics = build_fold(features, metadata, spec, frozen,
                                             final_holdout=spec.name == "holdout")
        if reference is not None:
            _, old_fit, _ = _reference_keys(reference, "fit", spec.name)
            if not old_fit["pkg_no"].isin(phase_keys["fit"]["pkg_no"]).all():
                raise ValueError("Deployment-C fit must contain every reference fitting key")
            for phase in ["cal", "eval"]:
                _, old_keys, _ = _reference_keys(reference, phase, spec.name)
                if _key_digest(phase_keys[phase]) != _key_digest(old_keys):
                    raise ValueError(f"Deployment-C {phase} keys differ from reference")
        diagnostics["artifacts"] = {phase: _write_keys_once(split_root / spec.name / f"{phase}_ids.parquet", ids)
                                    for phase, ids in phase_keys.items()}
        report["folds"][spec.name] = diagnostics
        _atomic_json(report_path, report)
    if reference is not None:
        report["reference"]["old_fit_is_subset"] = True
    report["status"] = "complete"
    report["holdout_isolation_checked"] = True
    _atomic_json(report_path, report)
    return report

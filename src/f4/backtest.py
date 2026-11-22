"""Guarded development backtesting with frozen cohorts and explicit authorization."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml

from .calibration import ResidualCalibrator, ShrunkResidualCalibrator
from .common import PROJECT_ROOT, atomic_write_json, project_path, sha256_file, timestamp
from .features import FeatureBuilder, SHARED_COLUMNS
from .metrics import score_predictions
from .models import QuantileBaseline, save_model, ITERATION_ALIASES
from .models import make_conditional_model, load_model
from .targets import TargetTransform
from .diagnostics import cohort_diagnostics, prediction_diagnostics
from .safety import TrainingNotAuthorizedError
from .splits import _key_digest, validate_config


def _safe_id(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"Invalid {field}")
    return value


def _authorize(config: dict, fold_name: str, execute_training: bool, synthetic: bool) -> None:
    execution = config.get("execution", {})
    if not synthetic and not (execute_training is True
                              and execution.get("stage") == "training"
                              and execution.get("allow_real_training") is True):
        raise TrainingNotAuthorizedError(
            "Real-data fitting is locked: explicit execution, training stage and allow_real_training are all required")
    if fold_name == "holdout" and (synthetic or not config.get("holdout_route_locked", False)):
        raise TrainingNotAuthorizedError("Final holdout requires a separately locked route and real-training authorization")
    if not synthetic and execution.get("allowed_folds") is not None and fold_name not in execution["allowed_folds"]:
        raise TrainingNotAuthorizedError("Requested fold is outside this configuration's authorized scope")
    if config.get("protocol", "asof_backtest") != "asof_backtest":
        raise ValueError("This entry point only implements asof_backtest")


def _official_score(truth: pd.DataFrame, predictions: pd.DataFrame) -> tuple[dict, str]:
    scorer = PROJECT_ROOT / "official/tools/score.py"
    spec = importlib.util.spec_from_file_location("f4_official_score_for_backtest", scorer)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    aligned = truth.merge(predictions, on="pkg_no", how="left", sort=False, validate="one_to_one")
    result = module.score(aligned["lgs_company_nm"], aligned["y_total_hours"], aligned["pred_hours"])
    return {key: float(value) for key, value in result.items()}, sha256_file(scorer)


def _load_keys(path: Path, recorded: dict, all_ids: pd.Series) -> pd.DataFrame:
    if sha256_file(path) != recorded.get("file_sha256"):
        raise ValueError(f"Frozen key artifact hash changed: {path}")
    keys = pd.read_parquet(path, columns=["pkg_no"], use_threads=False)
    if keys.empty or keys["pkg_no"].isna().any() or keys["pkg_no"].duplicated().any():
        raise ValueError(f"Invalid frozen keys: {path}")
    if not keys["pkg_no"].map(lambda value: isinstance(value, str) and bool(value)).all():
        raise ValueError("Frozen package keys must retain string semantics")
    if len(keys) != recorded.get("n") or _key_digest(keys) != recorded.get("keys_sha256"):
        raise ValueError(f"Frozen key contents disagree with split report: {path}")
    if not keys["pkg_no"].isin(all_ids).all():
        raise ValueError(f"Frozen keys are not all present in the source core table: {path}")
    return keys


def _validate_times(frame: pd.DataFrame, phase: str, times: dict) -> None:
    issue = pd.to_datetime(frame["issue_time"], errors="coerce")
    sign = pd.to_datetime(frame["sign_time"], errors="coerce")
    F, E, C = (pd.Timestamp(times[key]) for key in ("F", "E", "C"))
    end = C + pd.Timedelta(days=times["H_days"])
    valid = {"fit": issue.lt(F) & sign.lt(F),
             "cal": issue.ge(F) & issue.lt(E) & sign.lt(C),
             "eval": issue.ge(C) & issue.lt(end)}[phase]
    if not valid.all():
        raise ValueError(f"Frozen {phase} keys violate recorded issue/sign-time boundaries")
    if phase != "eval" and sign.isna().any():
        raise ValueError(f"{phase} contains an invalid available-label timestamp")


def _prepare_inputs(config: dict, fold_name: str, synthetic: bool) -> tuple[dict, dict[str, pd.DataFrame], dict]:
    data_dir = project_path(config.get("data_dir", "data/processed/core"))
    train_path = data_dir / "train.parquet"
    if synthetic and pq.ParquetFile(train_path).metadata.num_rows > 200:
        raise TrainingNotAuthorizedError("Synthetic smoke is limited to at most 200 total input rows")
    all_keys = pd.read_parquet(train_path, columns=["pkg_no"], use_threads=False)
    if all_keys.empty or all_keys["pkg_no"].isna().any() or all_keys["pkg_no"].duplicated().any():
        raise ValueError("Source core table must contain unique nonempty package keys")
    if synthetic and not all_keys["pkg_no"].astype("string").str.startswith("SMOKE_").all():
        raise TrainingNotAuthorizedError("Every source row must have a SMOKE_ package key")
    splits_path = project_path(config.get("splits", "configs/splits.yaml"))
    split_config = yaml.safe_load(splits_path.read_text(encoding="utf-8"))
    split_id, holdout_spec, development = validate_config(split_config)
    candidates = {spec.name: spec for spec in [*development, holdout_spec]}
    if fold_name not in candidates:
        raise ValueError(f"Unknown fold: {fold_name}")
    spec = candidates[fold_name]
    canonical = {"split_id": split_id, "holdout": holdout_spec.metadata(),
                 "development": [value.metadata() for value in development]}
    split_report_base = project_path(config.get("split_reports_base", str(project_path(config.get("report_base", "reports")) / "splits")))
    report_path = split_report_base / f"{split_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    canonical_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    train_hash = sha256_file(train_path)
    if report.get("status") != "complete" or report.get("holdout_isolation_checked") is not True:
        raise ValueError("Only complete, isolated frozen splits may be used")
    if (report.get("split_id") != split_id or report.get("config") != canonical
            or report.get("config_sha256") != canonical_hash
            or report.get("config_file_sha256") != sha256_file(splits_path)
            or report.get("train_parquet_sha256") != train_hash):
        raise ValueError("Frozen split source/configuration provenance does not match current inputs")
    fold_report = report["folds"][fold_name]
    if fold_report.get("times") != spec.metadata() or fold_report.get("protocol") != "asof_backtest":
        raise ValueError("Fold times/protocol disagree with the frozen split configuration")
    artifact_base = project_path(config.get("artifacts_base", "artifacts"))
    split_root = artifact_base / "splits" / split_id
    isolated = _load_keys(split_root / "holdout_ids.parquet", report["holdout"], all_keys["pkg_no"])
    keys = {phase: _load_keys(split_root / fold_name / f"{phase}_ids.parquet",
                              fold_report["artifacts"][phase], all_keys["pkg_no"])
            for phase in ("fit", "cal", "eval")}
    phases = list(keys)
    for position, phase in enumerate(phases):
        if (phase != "eval" or fold_name != "holdout") and keys[phase]["pkg_no"].isin(isolated["pkg_no"]).any():
            raise ValueError(f"{phase} includes isolated final-holdout keys")
        for other in phases[position + 1:]:
            if keys[phase]["pkg_no"].isin(keys[other]["pkg_no"]).any():
                raise ValueError(f"Frozen {phase}/{other} cohorts overlap")
    if fold_name == "holdout" and _key_digest(keys["eval"]) != _key_digest(isolated):
        raise ValueError("Final evaluation cohort differs from the isolated holdout")
    # Arrow applies the key filter before materializing label-bearing pandas rows;
    # development runs never materialize final-holdout labels in their cohorts.
    selected_ids = pd.concat(keys.values(), ignore_index=True)["pkg_no"].tolist()
    columns = [*SHARED_COLUMNS, "sign_time", "y_total_hours"]
    source = pd.read_parquet(train_path, columns=columns, filters=[("pkg_no", "in", selected_ids)], use_threads=False)
    cohorts = {phase: ids.merge(source, on="pkg_no", how="left", sort=False, validate="one_to_one")
               for phase, ids in keys.items()}
    for phase, frame in cohorts.items():
        if (len(frame) != len(keys[phase])
                or not frame["pkg_no"].astype("string").reset_index(drop=True).equals(
                    keys[phase]["pkg_no"].astype("string").reset_index(drop=True))):
            raise ValueError("Cohort join changed key order/count")
        # pandas 3 may merge Arrow/string backends into a different string dtype;
        # preserve key values/order while normalizing only the storage backend.
        frame["pkg_no"] = frame["pkg_no"].astype("string")
        _validate_times(frame, phase, spec.metadata())
    provenance = {"split_id": split_id, "split_report": str(report_path),
                  "split_report_sha256": sha256_file(report_path), "train_parquet_sha256": train_hash,
                  "split_config_sha256": canonical_hash, "times": spec.metadata(),
                  "cohort_keys": {phase: fold_report["artifacts"][phase] for phase in phases}}
    return provenance, cohorts, report


def run_backtest(config, fold_name, run_id, *, execute_training=False, synthetic=False):
    """Run an explicitly authorized backtest, or a <=200-row synthetic smoke.

    Preparation configs remain locked; local pilot configs authorize dev_1 only.
    The synthetic flag alone cannot fit real F4 rows: the entire input core file,
    not merely the chosen fold, must contain only SMOKE_ keys and <=200 rows.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a parsed configuration mapping")
    run_id, fold_name = _safe_id(run_id, "run_id"), _safe_id(fold_name, "fold_name")
    _authorize(config, fold_name, execute_training, synthetic)
    experiment = str(config.get("experiment", "")).upper()
    if experiment not in {"B0", "B1", "M1", "M2"}:
        raise ValueError("experiment must be B0, B1, M1 or M2")
    model_params = dict(config.get("model", {}))
    target = TargetTransform(config.get('target', 'total'))
    family = config.get('model_family', 'lightgbm')
    use_calibration = experiment == 'M2' or (
        experiment == 'B1' and config.get('calibration', {}).get('enabled') is True)
    if config.get('calibration', {}).get('enabled') is True and experiment not in {'B1', 'M2'}:
        raise ValueError('Explicit calibration is supported for B1 or M2 only')
    calibration_method = config.get('calibration', {}).get('method', 'independent_carrier_residual')
    if use_calibration and calibration_method not in {'independent_carrier_residual', 'shrunk_carrier_residual'}:
        raise ValueError('Unsupported independent calibration method')
    if calibration_method == 'shrunk_carrier_residual' and experiment != 'M2':
        raise ValueError('Shrinkage calibration is an M2 variant only')
    if config.get('reuse_model_run') and experiment not in {'M1', 'M2'}:
        raise ValueError('Only conditional models may reuse a frozen fitted model')
    if experiment in {'B0', 'B1'} and (target.kind != 'total' or family != 'lightgbm'):
        raise ValueError('B0/B1 remain total-hour baselines without alternate model families')
    if experiment in {"B0", "B1"}:
        expected_kind = {"global", "global_constant"} if experiment == "B0" else {"carrier", "carrier_constant"}
        if model_params.get("kind", next(iter(expected_kind))) not in expected_kind:
            raise ValueError("Baseline model kind disagrees with the experiment name")
        if list(model_params.get("quantiles", [.90, .97])) != [.90, .97]:
            raise ValueError("Baselines use the fixed 0.90/0.97 commitment rule")
    alphas = model_params.pop("alphas", [.90, .97])
    if experiment in {"M1", "M2"} and list(alphas) != [.90, .97]:
        raise ValueError("M1/M2 use the frozen 0.90 and 0.97 quantile objectives")
    if experiment in {'M1', 'M2'} and set(model_params).intersection(ITERATION_ALIASES):
        raise ValueError('Use n_estimators only; iteration aliases are not allowed')
    if synthetic and experiment in {"M1", "M2"}:
        if (not 1 <= model_params.get("n_estimators", 800) <= 5
                or model_params.get("n_jobs", 4) != 1):
            raise TrainingNotAuthorizedError("Synthetic models are limited to at most five rounds and one thread")
    conditional = make_conditional_model(family, model_params) if experiment in {'M1', 'M2'} else None
    locations = {"artifacts": project_path(config.get("artifacts_base", "artifacts")) / "runs" / run_id,
                 "reports": project_path(config.get("report_base", "reports")) / "runs" / run_id,
                 "outputs": project_path(config.get("output_base", "outputs")) / "runs" / run_id}
    if any(path.exists() for path in locations.values()):
        raise FileExistsError(f"run_id already exists; refusing overwrite: {run_id}")
    provenance, cohorts, split_report = _prepare_inputs(config, fold_name, synthetic)
    for path in locations.values():
        path.mkdir(parents=True, exist_ok=False)
    manifest_path = locations["reports"] / "run_manifest.json"
    manifest = {"run_id": run_id, "fold": fold_name, "experiment": experiment,
                "model_family": family, "target": target.kind, "calibration_enabled": use_calibration,
                "protocol": "asof_backtest", "synthetic": bool(synthetic),
                "not_competition_result": True,
                "evaluation_scope": "synthetic_smoke" if synthetic else (
                    "final_holdout_backtest" if fold_name == "holdout" else "development_backtest"),
                "status": "running", "started_at": timestamp(),
                "source": provenance, "config": config, "locations": {key: str(path) for key, path in locations.items()},
                "cohort_n": {phase: len(frame) for phase, frame in cohorts.items()},
                "artifact_format": "pickle: load only trusted local artifacts"}
    atomic_write_json(manifest_path, manifest)
    (locations["reports"] / "config.resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    try:
        fit, cal, evaluation = (cohorts[phase] for phase in ("fit", "cal", "eval"))
        cohort_report = locations['reports'] / 'cohort_diagnostics.json'
        atomic_write_json(cohort_report, cohort_diagnostics(
            cohorts, provenance['times'], split_report['folds'][fold_name].get('calibration_maturity')))
        manifest['cohort_diagnostics_sha256'] = sha256_file(cohort_report)
        fit_ids, cal_ids = fit["pkg_no"].tolist(), cal["pkg_no"].tolist()
        encoder = calibrator = None
        diagnostics = None
        if experiment in {"B0", "B1"}:
            model = QuantileBaseline(kind="global" if experiment == "B0" else "carrier",
                                     min_group_samples=model_params.get("min_group_samples", 200))
            model.fit(fit["y_total_hours"], fit["lgs_company_nm"], pkg_ids=fit_ids)
            values, diagnostics = model.predict(evaluation["lgs_company_nm"], return_diagnostics=True)
            if use_calibration:
                calibration_base = model.predict(cal['lgs_company_nm'])
        else:
            fitted_target = target.transform(fit)
            target.offset(cal)
            target.offset(evaluation)
            if config.get('reuse_model_run'):
                base_run = _safe_id(config['reuse_model_run'], 'reuse_model_run')
                base_manifest_path = project_path(config.get('report_base', 'reports')) / 'runs' / base_run / 'run_manifest.json'
                previous = json.loads(base_manifest_path.read_text(encoding='utf-8'))
                if (previous.get('status') != 'complete' or previous.get('synthetic') != bool(synthetic)
                        or previous.get('fold') != fold_name or previous.get('source') != provenance):
                    raise ValueError('Reused model must have identical source, scope and frozen cohorts')
                for field, default in [('model_family', 'lightgbm'), ('target', 'total'), ('features', 'shared_fields_v1')]:
                    if previous['config'].get(field, default) != config.get(field, default):
                        raise ValueError(f'Reused model {field} differs')
                if previous['config']['model'] != config['model']:
                    raise ValueError('Reused model parameters differ')
                for name in ['models.py', 'features.py', 'targets.py']:
                    if previous['code_files_sha256'][name] != sha256_file(Path(__file__).parent / name):
                        raise ValueError(f'Reused model component changed: {name}')
                loaded = {}
                for name in ['model', 'encoder']:
                    artifact = previous['artifacts'][name]
                    artifact_path = project_path(artifact['path']).resolve()
                    expected_dir = (locations['artifacts'].parent / base_run).resolve()
                    if not artifact_path.is_relative_to(expected_dir) or sha256_file(artifact_path) != artifact['sha256']:
                        raise ValueError(f'Reused {name} path or digest changed')
                    loaded[name] = load_model(artifact_path)
                model, encoder = loaded['model'], loaded['encoder']
                from .features import pkg_ids_sha256
                if (model.training_ids_sha256_ != pkg_ids_sha256(fit_ids)
                        or encoder.fit_ids_sha256_ != model.training_ids_sha256_
                        or encoder.encoder_sha256_ != model.feature_encoder_sha256_):
                    raise ValueError('Reused model/encoder are not bound to this fitting cohort')
                manifest['reused_model'] = {'run_id': base_run, 'manifest_sha256': sha256_file(base_manifest_path),
                                             'model_sha256': previous['artifacts']['model']['sha256']}
            else:
                encoder = FeatureBuilder()
                matrix = encoder.fit_transform(fit.loc[:, list(SHARED_COLUMNS)])
                model = conditional.fit(matrix, fitted_target, pkg_ids=fit_ids)
            values = target.restore(model.predict(encoder.transform(evaluation.loc[:, list(SHARED_COLUMNS)])), evaluation)
            if use_calibration:
                calibration_base = target.restore(model.predict(encoder.transform(cal.loc[:, list(SHARED_COLUMNS)])), cal)
        if use_calibration:
            cal_prediction_path = locations['outputs'] / 'calibration_base_predictions.csv'
            pd.DataFrame({'pkg_no': cal['pkg_no'], 'pred_hours': calibration_base}).to_csv(cal_prediction_path, index=False)
            calibrator_class = ShrunkResidualCalibrator if calibration_method == 'shrunk_carrier_residual' else ResidualCalibrator
            calibrator = calibrator_class(min_group_samples=config.get('calibration', {}).get('min_group_samples', 200))
            calibrator.fit(cal['y_total_hours'], calibration_base, cal['lgs_company_nm'],
                           fit_ids=fit_ids, cal_ids=cal_ids,
                           model_fit_ids_sha256=model.training_ids_sha256_)
            values, diagnostics = calibrator.apply(values, evaluation['lgs_company_nm'], return_diagnostics=True)
            manifest['calibration_base_predictions_sha256'] = sha256_file(cal_prediction_path)
            cal_truth = cal[['pkg_no', 'lgs_company_nm', 'y_total_hours']]
            cal_after = calibrator.apply(calibration_base, cal['lgs_company_nm'])
            calibration_report = locations['reports'] / 'calibration_diagnostics.json'
            atomic_write_json(calibration_report, {
                'scope': 'Calibration resubstitution diagnostics; not independent validation',
                'before': prediction_diagnostics(cal_truth, pd.DataFrame({'pkg_no': cal['pkg_no'], 'pred_hours': calibration_base})),
                'after': prediction_diagnostics(cal_truth, pd.DataFrame({'pkg_no': cal['pkg_no'], 'pred_hours': cal_after}))})
            manifest['calibration_diagnostics_sha256'] = sha256_file(calibration_report)
        saved = {}
        for name, artifact in [("model", model), ("encoder", encoder), ("calibrator", calibrator)]:
            if artifact is not None:
                path = locations["artifacts"] / f"{name}.pkl"
                save_model(artifact, path)
                saved[name] = {"path": str(path), "sha256": sha256_file(path)}
        if encoder is not None:
            atomic_write_json(locations["artifacts"] / "encoder.json", encoder.to_dict())
        prediction_path = locations["outputs"] / "predictions.csv"
        pd.DataFrame({"pkg_no": evaluation["pkg_no"], "pred_hours": values}).to_csv(prediction_path, index=False)
        persisted = pd.read_csv(prediction_path, dtype={"pkg_no": "string"})
        truth = evaluation[["pkg_no", "lgs_company_nm", "y_total_hours"]]
        metrics, groups = score_predictions(truth, persisted)
        prediction_report = locations['reports'] / 'prediction_diagnostics.json'
        atomic_write_json(prediction_report, prediction_diagnostics(truth, persisted))
        manifest['prediction_diagnostics_sha256'] = sha256_file(prediction_report)
        official, official_sha = _official_score(truth, persisted)
        if any(abs(metrics[key] - official[key]) >= 1e-8 for key in metrics):
            raise ValueError("Local diagnostics and unmodified official scorer disagree")
        if diagnostics is not None:
            detail = diagnostics.drop_duplicates(subset=["group"])
            groups = groups.merge(detail, on="group", how="left", validate="one_to_one")
        groups.to_csv(locations["reports"] / "by_carrier.csv", index=False)
        atomic_write_json(locations["reports"] / "metrics.json",
                          {"synthetic": bool(synthetic), "not_competition_result": True,
                           "evaluation_scope": manifest['evaluation_scope'], **metrics})
        source_files = ["backtest.py", "models.py", "features.py", "targets.py", "platform_utils.py", "calibration.py", "diagnostics.py", "metrics.py", "splits.py", "safety.py", "common.py"]
        manifest.update(status="complete", finished_at=timestamp(), metrics=metrics,
                        official_metrics=official, official_scorer_sha256=official_sha,
                        artifacts=saved, predictions={"path": str(prediction_path), "sha256": sha256_file(prediction_path)},
                        model_fit_ids_sha256=model.training_ids_sha256_,
                        code_files_sha256={name: sha256_file(Path(__file__).parent / name) for name in source_files})
        if calibrator is not None:
            manifest["calibration_fit_ids_sha256"] = calibrator.fit_ids_sha256_
            manifest["calibration_ids_sha256"] = calibrator.cal_ids_sha256_
            manifest["calibration_base_model_sha256"] = saved["model"]["sha256"]
        atomic_write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest.update(status="failed", finished_at=timestamp(), error=f"{type(exc).__name__}: {exc}")
        atomic_write_json(manifest_path, manifest)
        raise

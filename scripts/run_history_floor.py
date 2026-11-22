"""One fixed development-only mature-history floor experiment, no search."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml

from f4.backtest import _authorize, _official_score, _safe_id
from f4.common import atomic_write_json, sha256_file, source_snapshot, timestamp
from f4.diagnostics import prediction_diagnostics
from f4.features import pkg_ids_sha256, validate_pkg_ids
from f4.history_floor import apply_history_floor, fit_history_floor
from f4.platform_utils import current_process_peak_memory, lock_file
from f4.splits import _key_digest, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    batch_id = _safe_id(args.run_id, 'run-id')
    config_path = ROOT / 'configs/experiments/mature_history_floor.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    for fold in ('dev_1', 'dev_2'):
        _authorize(config, fold, args.execute_training, False)
    if config['history'] != {'min_age_days': 21, 'min_group_samples': 200,
                             'label_cutoff': 'C', 'rule': 'max_Q90_Q97_minus_24'}:
        raise ValueError('This experiment permits only the predeclared fixed recipe')
    locations = {kind: ROOT / kind / 'runs' / batch_id for kind in ('reports', 'artifacts')}
    run_ids = [f'{batch_id}_{fold}_{variant}' for fold in ('dev_1', 'dev_2') for variant in ('carrier', 'lightgbm')]
    if any(p.exists() for p in locations.values()) or any((ROOT / k / 'runs' / r).exists()
            for r in run_ids for k in ('reports', 'outputs', 'artifacts')):
        raise FileExistsError('Experiment IDs cannot be overwritten')
    started = time.perf_counter()
    snapshots, snapshot_hash = source_snapshot()
    core = ROOT / 'data/processed/core/train.parquet'
    splits_path = ROOT / config['splits']
    split_id, holdout_spec, development = validate_config(yaml.safe_load(splits_path.read_text(encoding='utf-8')))
    split_report_path = ROOT / 'reports/splits' / (split_id + '.json')
    split = json.loads(split_report_path.read_text(encoding='utf-8'))
    expected_config = {'split_id': split_id, 'holdout': holdout_spec.metadata(),
                       'development': [s.metadata() for s in development]}
    if (split['status'] != 'complete' or not split['holdout_isolation_checked']
            or split['config'] != expected_config
            or split['config_file_sha256'] != sha256_file(splits_path)
            or split['train_parquet_sha256'] != sha256_file(core)):
        raise ValueError('Frozen source or split changed')
    verified = json.loads((ROOT / 'reports/local_pilot/catboost_round5_20261118/comparison.json').read_text(encoding='utf-8'))
    reference_hashes = {r['run_id']: r['manifest_sha256'] for r in verified['runs']}

    def keys(record):
        path = Path(record['path'])
        if sha256_file(path) != record['file_sha256']:
            raise ValueError('Frozen key file changed')
        frame = pd.read_parquet(path, columns=['pkg_no'], use_threads=False)
        validate_pkg_ids(frame.pkg_no, len(frame))
        if len(frame) != record['n'] or _key_digest(frame) != record['keys_sha256']:
            raise ValueError('Frozen key contents changed')
        return frame

    holdout = keys(split['holdout'])
    dataset = ds.dataset(core, format='parquet')
    issue_type = dataset.schema.field('issue_time').type
    sign_type = dataset.schema.field('sign_time').type
    for path in locations.values():
        path.mkdir(parents=True, exist_ok=False)
    atomic_write_json(locations['reports'] / 'code_snapshot.json', snapshots)
    result = {'status': 'running', 'started_at': timestamp(), 'run_id': batch_id,
              'evaluation_scope': 'development_backtest', 'not_competition_result': True,
              'final_holdout_evaluated': False, 'parameter_search_performed': False,
              'base_models_retrained': False, 'config': config,
              'config_sha256': sha256_file(config_path), 'source_snapshot_sha256': snapshot_hash,
              'source': {'train_parquet_sha256': split['train_parquet_sha256'],
                         'split_report_sha256': sha256_file(split_report_path)},
              'history': {}, 'runs': []}
    report_path = locations['reports'] / 'summary.json'
    atomic_write_json(report_path, result)
    try:
        with (ROOT / 'reports/local_pilot.lock').open('a') as lock:
            lock_file(lock)
            for fold_name in ('dev_1', 'dev_2'):
                fold = split['folds'][fold_name]
                times = fold['times']
                cutoff = pd.Timestamp(times['C'])
                age_cutoff = cutoff - pd.Timedelta(days=21)
                # Only decision-time keys are read before holdout intersection check.
                history_keys = dataset.to_table(columns=['pkg_no'], filter=(
                    ds.field('issue_time') < pa.scalar(age_cutoff, type=issue_type)), use_threads=False).to_pandas()
                validate_pkg_ids(history_keys.pkg_no, len(history_keys))
                evaluation_keys = keys(fold['artifacts']['eval'])
                if (history_keys.pkg_no.isin(holdout.pkg_no).any()
                        or evaluation_keys.pkg_no.isin(holdout.pkg_no).any()
                        or history_keys.pkg_no.isin(evaluation_keys.pkg_no).any()):
                    raise ValueError('History/evaluation/holdout isolation failed')
                history = dataset.to_table(columns=['pkg_no', 'lgs_company_nm', 'issue_time', 'sign_time', 'y_total_hours'],
                    filter=(ds.field('pkg_no').isin(history_keys.pkg_no.tolist())
                            & (ds.field('sign_time') < pa.scalar(cutoff, type=sign_type))), use_threads=False).to_pandas()
                fitted = fit_history_floor(history, cutoff, 21, 200)
                history_path = locations['artifacts'] / (fold_name + '_history_ids.parquet')
                history[['pkg_no']].to_parquet(history_path, index=False)
                floor_path = locations['artifacts'] / (fold_name + '_floor.json')
                atomic_write_json(floor_path, fitted)
                fit_keys = keys(fold['artifacts']['fit'])
                cal_keys = keys(fold['artifacts']['cal'])
                if history.pkg_no.isin(cal_keys.pkg_no).any():
                    raise ValueError('Older history unexpectedly intersects original calibration queue')
                info = {'fitted': fitted, 'history_ids_path': str(history_path),
                        'history_ids_sha256': sha256_file(history_path),
                        'floor_path': str(floor_path), 'floor_sha256': sha256_file(floor_path),
                        'candidate_queue_n': len(history_keys),
                        'visible_fraction_within_official_queue': len(history) / len(history_keys),
                        'overlap_original_fit_n': int(history.pkg_no.isin(fit_keys.pkg_no).sum()),
                        'overlap_original_cal_n': 0, 'overlap_eval_n': 0, 'overlap_holdout_n': 0,
                        'fitted_at': timestamp()}
                result['history'][fold_name] = info
                atomic_write_json(report_path, result)
                del history, history_keys, fit_keys, cal_keys
                # Read prediction-time features only; fit/freeze floor before evaluation labels.
                features = dataset.to_table(columns=['pkg_no', 'lgs_company_nm'],
                    filter=ds.field('pkg_no').isin(evaluation_keys.pkg_no.tolist()), use_threads=False).to_pandas()
                aligned = evaluation_keys.merge(features, on='pkg_no', how='left', sort=False,
                                               validate='one_to_one', indicator=True)
                if not aligned['_merge'].eq('both').all() or aligned.pkg_no.tolist() != evaluation_keys.pkg_no.tolist():
                    raise ValueError('Evaluation feature join mismatch')
                pending = []
                for variant in ('carrier', 'lightgbm'):
                    reference_id = config['reference_runs'][fold_name][variant]
                    original_path = ROOT / 'reports/runs' / reference_id / 'run_manifest.json'
                    if sha256_file(original_path) != reference_hashes[reference_id]:
                        raise ValueError('Original manifest changed')
                    original = json.loads(original_path.read_text(encoding='utf-8'))
                    if (original['status'] != 'complete' or original['synthetic'] or original['fold'] != fold_name
                            or original['source']['train_parquet_sha256'] != split['train_parquet_sha256']
                            or original['source']['split_report_sha256'] != sha256_file(split_report_path)
                            or original['source']['cohort_keys'] != fold['artifacts']):
                        raise ValueError('Original run source mismatch')
                    prediction_path = Path(original['predictions']['path'])
                    if sha256_file(prediction_path) != original['predictions']['sha256']:
                        raise ValueError('Original predictions changed')
                    base = pd.read_csv(prediction_path, dtype={'pkg_no': 'string'})
                    validate_pkg_ids(base.pkg_no, len(base))
                    if base.pkg_no.tolist() != evaluation_keys.pkg_no.tolist():
                        raise ValueError('Original prediction keys/order differ from frozen evaluation')
                    values = apply_history_floor(base.pred_hours, aligned.lgs_company_nm, fitted)
                    run_id = f'{batch_id}_{fold_name}_{variant}'
                    report_dir, out_dir = (ROOT / p / 'runs' / run_id for p in ('reports', 'outputs'))
                    report_dir.mkdir(parents=True, exist_ok=False)
                    out_dir.mkdir(parents=True, exist_ok=False)
                    predictions = pd.DataFrame({'pkg_no': base.pkg_no, 'pred_hours': values})
                    path = out_dir / 'predictions.csv'
                    predictions.to_csv(path, index=False)
                    run = {'status': 'predicted', 'run_id': run_id, 'fold': fold_name, 'variant': variant,
                           'reference_run': reference_id, 'reference_manifest_sha256': sha256_file(original_path),
                           'reference_prediction_sha256': original['predictions']['sha256'],
                           'reference_metrics': original['metrics'], 'source': original['source'],
                           'cohort_n': original['cohort_n'], 'config_sha256': result['config_sha256'],
                           'method': fitted['method'], 'history_artifact': info,
                           'base_model_retrained': False, 'parameter_search_performed': False,
                           'evaluation_scope': 'development_backtest', 'not_competition_result': True,
                           'final_holdout_evaluated': False, 'synthetic': False,
                           'predictions': {'path': str(path), 'sha256': sha256_file(path)},
                           'predictions_frozen_at': timestamp(),
                           'raised_prediction_n': int((values > base.pred_hours.to_numpy()).sum()),
                           'mean_increase_hours': float((values - base.pred_hours.to_numpy()).mean())}
                    atomic_write_json(report_dir / 'run_manifest.json', run)
                    pending.append((run, report_dir, predictions, base))
                # Evaluation labels are materialized only after both variant predictions are frozen.
                truth = dataset.to_table(columns=['pkg_no', 'lgs_company_nm', 'y_total_hours'],
                    filter=ds.field('pkg_no').isin(evaluation_keys.pkg_no.tolist()), use_threads=False).to_pandas()
                for run, folder, predictions, base in pending:
                    diagnostic = prediction_diagnostics(truth, predictions)
                    original_metrics, _ = _official_score(truth, base)
                    if any(abs(original_metrics[k] - run['reference_metrics'][k]) > 1e-8 for k in original_metrics):
                        raise ValueError('Reference score cannot be reproduced')
                    official, scorer_hash = _official_score(truth, predictions)
                    error = max(abs(official[k] - diagnostic['metrics'][k]) for k in official)
                    if error > 1e-8:
                        raise ValueError('Official scorer mismatch')
                    atomic_write_json(folder / 'prediction_diagnostics.json', diagnostic)
                    run.update(status='complete', finished_at=timestamp(), metrics=diagnostic['metrics'],
                               official_metrics=official, official_scorer_sha256=scorer_hash,
                               official_score_max_abs_error=error,
                               prediction_diagnostics_sha256=sha256_file(folder / 'prediction_diagnostics.json'),
                               source_snapshot_sha256=snapshot_hash,
                               improvement_A=run['reference_metrics']['A'] - diagnostic['metrics']['A'])
                    atomic_write_json(folder / 'run_manifest.json', run)
                    result['runs'].append({'run_id': run['run_id'], 'fold': fold_name, 'variant': run['variant'],
                                           'manifest': str(folder / 'run_manifest.json'),
                                           'manifest_sha256': sha256_file(folder / 'run_manifest.json'),
                                           'metrics': run['metrics'], 'reference_metrics': run['reference_metrics'],
                                           'improvement_A': run['improvement_A']})
                    atomic_write_json(report_path, result)
                    print(f"Completed {run['run_id']}: A={run['metrics']['A']:.6f}", flush=True)
                del pending, truth, features, aligned, evaluation_keys
            if source_snapshot()[1] != snapshot_hash:
                raise ValueError('Source changed during experiment')
            result.update(status='complete', finished_at=timestamp(),
                          elapsed_seconds=round(time.perf_counter() - started, 3),
                          resources=current_process_peak_memory(),
                          resource_scope='Whole serial postprocessing batch; includes history fit, provenance checks and scoring, excludes original tree training')
            atomic_write_json(report_path, result)
            print(report_path)
    except BaseException as exc:
        result.update(status='failed', finished_at=timestamp(), error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(report_path, result)
        raise


if __name__ == '__main__':
    main()

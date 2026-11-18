"""Retrospective development diagnostics; never fits models or reads holdout labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import yaml
from f4.common import atomic_write_json, project_path, sha256_file, timestamp
from f4.metrics import validate_predictions
from f4.splits import _checked_feature_masks, _key_digest, normalize_features, validate_config

FEATURE_COLUMNS = ['pkg_no', 'pay_time', 'issue_time', 'lgs_company_nm']
LABEL_COLUMNS = ['pkg_no', 'sign_time', 'y_total_hours']


def load_keys(path, record):
    path = Path(path)
    if sha256_file(path) != record['file_sha256']:
        raise ValueError(f'Frozen key file changed: {path}')
    keys = pd.read_parquet(path, columns=['pkg_no'], use_threads=False)
    if (keys['pkg_no'].isna().any() or keys['pkg_no'].duplicated().any()
            or len(keys) != record['n'] or _key_digest(keys) != record['keys_sha256']):
        raise ValueError(f'Frozen key contents changed: {path}')
    return keys


def load_development(data_dir, split_config_path, split_report_path, split_root,
                     *, reader=None):
    """Select allowed keys from features before materializing any labels."""
    reader = reader or pd.read_parquet
    train_path = Path(data_dir) / 'train.parquet'
    split_config_path, split_report_path = map(Path, [split_config_path, split_report_path])
    config = yaml.safe_load(split_config_path.read_text(encoding='utf-8'))
    split_id, holdout_spec, specs = validate_config(config)
    specs = [spec for spec in specs if spec.name in {'dev_1', 'dev_2'}]
    if {spec.name for spec in specs} != {'dev_1', 'dev_2'}:
        raise ValueError('Both frozen development windows are required')
    report = json.loads(split_report_path.read_text(encoding='utf-8'))
    canonical = {'split_id': split_id, 'holdout': holdout_spec.metadata(),
                 'development': [spec.metadata() for spec in validate_config(config)[2]]}
    config_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    source_hash = sha256_file(train_path)
    if (report.get('status') != 'complete' or report.get('holdout_isolation_checked') is not True
            or report.get('split_id') != split_id or report.get('config') != canonical
            or report.get('config_sha256') != config_hash
            or report.get('config_file_sha256') != sha256_file(split_config_path)
            or report.get('train_parquet_sha256') != source_hash):
        raise ValueError('Frozen split/source provenance does not match')
    split_root = Path(split_root)
    holdout = load_keys(split_root / 'holdout_ids.parquet', report['holdout'])
    features = reader(train_path, columns=FEATURE_COLUMNS, use_threads=False)
    normalized = normalize_features(features)
    features = features.copy()
    for column in ['pkg_no', 'pay_time', 'issue_time']:
        features[column] = normalized[column]
    if not holdout['pkg_no'].isin(features['pkg_no']).all():
        raise ValueError('Holdout keys are not present in the source')
    expected_holdout = features.loc[features['issue_time'].ge(holdout_spec.C)
                                    & features['issue_time'].lt(holdout_spec.end), ['pkg_no']]
    if _key_digest(expected_holdout) != _key_digest(holdout):
        raise ValueError('Holdout keys disagree with the frozen time window')
    cohorts, frozen, candidates = {}, {}, pd.Series(False, index=features.index)
    for spec in specs:
        masks, isolated = _checked_feature_masks(features, spec, holdout, False)
        fold = report['folds'][spec.name]
        if fold['times'] != spec.metadata():
            raise ValueError('Development times disagree with frozen report')
        frozen[spec.name] = {
            phase: load_keys(split_root / spec.name / f'{phase}_ids.parquet', entry)
            for phase, entry in fold['artifacts'].items()}
        cohorts[spec.name] = {}
        for phase, mask in masks.items():
            if (mask & isolated).any():
                raise ValueError('A development queue overlaps holdout')
            cohorts[spec.name][phase] = features.loc[mask, FEATURE_COLUMNS].copy()
            candidates |= mask
        if _key_digest(cohorts[spec.name]['eval'][['pkg_no']]) != _key_digest(frozen[spec.name]['eval']):
            raise ValueError('Evaluation queue differs from frozen keys')
    selected = features.loc[candidates, 'pkg_no'].tolist()
    if not selected or set(selected).intersection(holdout['pkg_no']):
        raise ValueError('Invalid diagnostic label selection')
    # The filter is part of the read: isolated labels are never pandas rows.
    labels = reader(train_path, columns=LABEL_COLUMNS,
                    filters=[('pkg_no', 'in', selected)], use_threads=False)
    if (labels['pkg_no'].isna().any() or labels['pkg_no'].duplicated().any()
            or set(labels['pkg_no']) != set(selected)):
        raise ValueError('Filtered labels do not exactly match allowed keys')
    labels = labels.copy()
    labels['pkg_no'] = labels['pkg_no'].astype('string')
    labels['sign_time'] = pd.to_datetime(labels['sign_time'], errors='raise')
    labels['y_total_hours'] = pd.to_numeric(labels['y_total_hours'], errors='raise')
    if (labels['sign_time'].isna().any()
            or not (np.isfinite(labels['y_total_hours']) & labels['y_total_hours'].gt(0)).all()):
        raise ValueError('Invalid development labels')
    for spec in specs:
        for phase, frame in cohorts[spec.name].items():
            joined = frame.merge(labels, on='pkg_no', how='left', sort=False, validate='one_to_one')
            joined['pkg_no'] = joined['pkg_no'].astype('string')
            if not joined['pkg_no'].reset_index(drop=True).equals(frame['pkg_no'].reset_index(drop=True)):
                raise ValueError('Diagnostic join changed key order')
            if phase != 'eval':
                cutoff = spec.F if phase == 'fit' else spec.C
                available = joined.loc[joined['sign_time'].lt(cutoff), ['pkg_no']]
                if _key_digest(available) != _key_digest(frozen[spec.name][phase]):
                    raise ValueError('Available diagnostic queue differs from frozen fitting/calibration keys')
            cohorts[spec.name][phase] = joined
    provenance = {'split_id': split_id, 'train_parquet_sha256': source_hash,
                  'split_config_sha256': sha256_file(split_config_path),
                  'split_report_sha256': sha256_file(split_report_path),
                  'holdout_keys_sha256': report['holdout']['keys_sha256'],
                  'candidate_label_rows': len(selected), 'filtered_label_rows': len(labels),
                  'excluded_holdout_rows': len(holdout)}
    return specs, cohorts, provenance


def duration_summary(values, prefix):
    values = np.asarray(values, dtype='float64')
    if not len(values):
        return {prefix + name: None for name in ['mean_hours', 'Q90_hours', 'Q97_hours']}
    return {prefix + 'mean_hours': float(values.mean()),
            **{prefix + f'Q{int(q * 100)}_hours': float(np.partition(values, math.ceil(q * len(values)) - 1)[math.ceil(q * len(values)) - 1])
               for q in [.90, .97]}}


def maturity_rows(frame, fold, phase, cutoff):
    """Full queue is the denominator; future labels are retrospective only."""
    work = frame.copy()
    work['issue_date'] = work['issue_time'].dt.strftime('%Y-%m-%d')
    result = []
    for (day, carrier), group in work.groupby(['issue_date', 'lgs_company_nm'], dropna=False, observed=True):
        available = group['sign_time'].lt(cutoff)
        result.append({'fold': fold, 'phase': phase, 'issue_date': day, 'carrier': carrier,
                       'cutoff_time': pd.Timestamp(cutoff).isoformat(), 'queue_n': len(group),
                       'available_n': int(available.sum()), 'unavailable_n': int((~available).sum()),
                       'available_rate': float(available.mean()),
                       **duration_summary(group.loc[available, 'y_total_hours'], 'available_'),
                       **duration_summary(group['y_total_hours'], 'retrospective_full_')})
    return result


def feature_bias_rows(frame, predictions, fold, model):
    aligned = validate_predictions(frame.reset_index(drop=True), predictions)
    elapsed = (aligned['issue_time'] - aligned['pay_time']).dt.total_seconds() / 3600
    if not (np.isfinite(elapsed) & elapsed.ge(0)).all():
        raise ValueError('Elapsed-time buckets require finite nonnegative values')
    factors = {'pay_weekday': aligned['pay_time'].dt.day_of_week.astype(str),
               'issue_weekday': aligned['issue_time'].dt.day_of_week.astype(str),
               'elapsed_hours': pd.cut(elapsed, [0, 1, 6, 24, np.inf], right=False,
                                       labels=['<1h', '1–6h', '6–24h', '>=24h'])}
    rows = []
    for feature, buckets in factors.items():
        work = aligned.assign(bucket=buckets)
        for bucket, group in work.groupby('bucket', observed=True, dropna=False):
            residual = group['y_total_hours'] - group['pred_hours']
            rows.append({'fold': fold, 'model': model, 'feature': feature, 'bucket': str(bucket),
                         'n': len(group), 'mean_pred_hours': float(group['pred_hours'].mean()),
                         'mean_true_hours': float(group['y_total_hours'].mean()),
                         'ontime_rate': float(residual.le(0).mean()),
                         'severe_rate': float(residual.gt(24).mean())})
    return rows


def diagnose(data_dir, split_config_path, split_report_path, split_root, batch_status_path,
             out, report_base=ROOT / 'reports/runs'):
    started = time.monotonic()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite diagnostic directory: {out}')
    specs, cohorts, provenance = load_development(data_dir, split_config_path, split_report_path, split_root)
    maturity = []
    for spec in specs:
        for phase in ['fit', 'cal']:
            maturity.extend(maturity_rows(cohorts[spec.name][phase], spec.name, phase,
                                          spec.F if phase == 'fit' else spec.C))
    status = json.loads(Path(batch_status_path).read_text(encoding='utf-8'))
    if status.get('status') != 'complete':
        raise ValueError('Diagnostics require a completed first-pass batch')
    bias, prediction_sources, seen = [], [], set()
    for job in status['jobs']:
        manifest_path = Path(report_base) / job['run_id'] / 'run_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('experiment') not in {'M1', 'M2'}:
            continue
        fold, model = manifest['fold'], manifest['experiment']
        if (fold not in cohorts or manifest.get('status') != 'complete' or manifest.get('synthetic') is not False
                or manifest['source']['split_id'] != provenance['split_id']
                or manifest['source']['train_parquet_sha256'] != provenance['train_parquet_sha256']
                or manifest['source']['split_report_sha256'] != provenance['split_report_sha256']
                or (fold, model) in seen):
            raise ValueError('Prediction provenance does not match development diagnostics')
        seen.add((fold, model))
        prediction = manifest['predictions']
        if sha256_file(prediction['path']) != prediction['sha256']:
            raise ValueError('Saved prediction file changed')
        predictions = pd.read_csv(prediction['path'], dtype={'pkg_no': 'string'})
        bias.extend(feature_bias_rows(cohorts[fold]['eval'], predictions, fold, model))
        prediction_sources.append({'fold': fold, 'model': model, 'run_id': job['run_id'],
                                   'manifest_sha256': sha256_file(manifest_path), **prediction})
    if seen != {(fold, model) for fold in ['dev_1', 'dev_2'] for model in ['M1', 'M2']}:
        raise ValueError('Both M1/M2 predictions are required for both development windows')
    out.mkdir(parents=True, exist_ok=False)
    maturity_path, bias_path = out / 'maturity_by_date_carrier.csv', out / 'feature_bias.csv'
    pd.DataFrame(maturity).to_csv(maturity_path, index=False)
    pd.DataFrame(bias).to_csv(bias_path, index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summary = {'created_at': timestamp(), 'status': 'complete', 'retrospective_diagnostic_only': True,
               'holdout_labels_read': False, 'training_performed': False,
               'interpretation': 'Full-label statistics are retrospective diagnostics, not historically available fitting/calibration parameters. Weekday 0 is Monday.',
               'source': provenance, 'batch_status_sha256': sha256_file(batch_status_path),
               'predictions': prediction_sources, 'elapsed_seconds': round(time.monotonic() - started, 3),
               'peak_RSS_GiB': round((rss if sys.platform == 'darwin' else rss * 1024) / 1024 ** 3, 4),
               'outputs': {p.name: sha256_file(p) for p in [maturity_path, bias_path]}}
    atomic_write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='reports/diagnostics/second_pass_20261118')
    parser.add_argument('--batch-status', default='reports/training/first_pass_20261118_v1/status.json')
    args = parser.parse_args()
    config_path = ROOT / 'configs/splits.yaml'
    split_id = yaml.safe_load(config_path.read_text())['split_id']
    result = diagnose(ROOT / 'data/processed/core', config_path,
                      ROOT / 'reports/splits' / f'{split_id}.json',
                      ROOT / 'artifacts/splits' / split_id,
                      project_path(args.batch_status), project_path(args.out))
    print(json.dumps({key: result[key] for key in ['status', 'retrospective_diagnostic_only',
                                                  'holdout_labels_read', 'elapsed_seconds', 'peak_RSS_GiB']}))


if __name__ == '__main__':
    main()

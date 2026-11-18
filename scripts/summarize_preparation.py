"""Write a verified preparation-only status; never fits or scores a model."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from f4.common import atomic_write_json, configuration_hash, sha256_file, timestamp, source_snapshot


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    prepared = read('data/processed/core/preparation_manifest.json')
    structure = read('reports/audit_structure.json')
    development = read('reports/audit_development.json')
    splits = read('reports/splits/time_v1_20261117.json')
    environment = read('reports/environment_check.json')
    smoke = read('reports/smoke/smoke_results.json')
    if not (prepared['status'] == 'complete' and splits['status'] == 'complete'
            and environment['status'] == 'ready' and smoke['passed']
            and not smoke['skipped'] and splits['holdout_isolation_checked']):
        raise ValueError('Preparation prerequisite or smoke check has not passed')
    code_hashes, snapshot_hash = source_snapshot()
    if smoke.get('source_snapshot_sha256') != snapshot_hash:
        raise ValueError('Source/configuration changed after the smoke test; rerun it before summarizing')
    for split, entry in prepared['files'].items():
        if sha256_file(entry['source_path']) != entry['source_sha256']:
            raise ValueError(f'Original {split} CSV changed')
        if sha256_file(ROOT / 'data/processed/core' / entry['parquet_path']) != entry['parquet_sha256']:
            raise ValueError(f'{split} cache changed')
    original_score = next((ROOT / 'data/raw/official').rglob('score.py'))
    scorer_sha = sha256_file(ROOT / 'official/tools/score.py')
    if scorer_sha != sha256_file(original_score):
        raise ValueError('Official scorer copy differs from original')
    configs = list((ROOT / 'configs/experiments').glob('*.yaml'))
    for path in configs:
        execution = yaml.safe_load(path.read_text())['execution']
        if execution['stage'] != 'preparation' or execution['allow_real_training'] is not False:
            raise ValueError('A real-training config was enabled during preparation')
    models = list((ROOT / 'artifacts/runs').glob('**/*.pkl'))
    predictions = list((ROOT / 'outputs/runs').glob('**/*.csv'))
    real_runs = list((ROOT / 'reports/runs').glob('*/run_manifest.json'))
    if models or predictions or real_runs:
        raise ValueError('Preparation-only summary found run artifacts; inspect before claiming no formal training')
    resources = {name: read(f'reports/preparation/{name}_resources.json')
                 for name in ['audit_structure', 'make_splits', 'audit_development']}
    if any(record['exit_code'] != 0 for record in resources.values()):
        raise ValueError('A full-data preparation stage failed')
    preflights = []
    for path in sorted((ROOT / 'reports/preparation').glob('preflight_20261117_complete_*.json')):
        report = json.loads(path.read_text())
        if not report['inputs_ready'] or not report.get('frozen_sources_verified'):
            raise ValueError('An input/source preflight failed')
        preflights.append(str(path.relative_to(ROOT)))
    if len(preflights) != 6:
        raise ValueError('Expected B1/M1/M2 dry runs for both development folds')
    atomic_write_json(ROOT / 'reports/preparation/code_snapshot.json', {
        'created_at': timestamp(), 'files_sha256': code_hashes,
        'snapshot_sha256': configuration_hash(code_hashes), 'version_control': 'source-file SHA256 snapshot',
    })
    status = {
        'checked_at': timestamp(), 'status': 'prepared_no_formal_training',
        'formal_training_started': False, 'real_validation_scores_produced': False,
        'environment_ready': True, 'official_scorer_unchanged': True,
        'official_scorer_sha256': scorer_sha, 'original_csv_hashes_verified': True,
        'data': {split: {'rows': record['rows'], 'raw_columns': record['raw_columns'],
                         'core_columns': len(record['columns']), 'cache_bytes': record['parquet_bytes'],
                         'cache_sha256': record['parquet_sha256']}
                 for split, record in prepared['files'].items()},
        'split_id': splits['split_id'], 'final_holdout_rows': splits['holdout']['n'],
        'development_audit_rows': development['development_rows'],
        'holdout_label_statistics_emitted': development['holdout_label_statistics_emitted'],
        'development_label_time_consistency': development['label_time_consistency'],
        'folds': {name: {'segments': {phase: segment['n'] for phase, segment in fold['segments'].items()},
                         'calibration_mature_fraction': fold['calibration_maturity']['mature_fraction']}
                  for name, fold in splits['folds'].items()},
        'shared_feature_anomalies': {split: record['anomalies'] for split, record in structure['datasets'].items()},
        'new_categories_in_test': {key: value for key, value in structure['category_coverage'].items()
                                   if value['test_unknown_rows']},
        'synthetic_smoke': {'tests': smoke['tests_run'], 'passed': smoke['passed'],
                            'skipped': len(smoke['skipped']), 'report': 'reports/smoke/smoke_results.json'},
        'dry_run_reports': preflights, 'full_data_resources': resources,
        'max_observed_preparation_RSS_GiB': round(max(r['peak_RSS_MiB'] for r in resources.values()) / 1024, 3),
        'profiling_note': 'Core conversion completed in about 50s; macOS time -l could not read a sandboxed sysctl. RSS above is from successful Python resource measurements on audit/split processes.',
        'code_snapshot_sha256': configuration_hash(code_hashes),
        'known_limitations': [
            'Calibration queues are not fully mature; signed-only filtering can bias tail calibration.',
            'Development label summaries are retrospective, not evidence of historical parameter availability.',
            'No real F4 fit, official validation result, final offline fit or test prediction has been run.',
            'Payment-cohort stress checks and M3 history features remain later-stage work.',
            'Final offline model/predict/submission packaging remains later-stage work.',
        ],
    }
    atomic_write_json(ROOT / 'reports/preparation_status.json', status)
    print(json.dumps({'status': status['status'], 'smoke_tests': smoke['tests_run'],
                      'holdout_rows': status['final_holdout_rows'],
                      'max_preparation_RSS_GiB': status['max_observed_preparation_RSS_GiB']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

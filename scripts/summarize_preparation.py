"""Validate a completed preparation pipeline against current local artifacts."""
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from f4.common import atomic_write_json, sha256_file, source_snapshot, timestamp


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True)
    parser.add_argument('--smoke', default='reports/smoke/smoke_results.json')
    args = parser.parse_args()
    pipeline_path = ROOT / args.pipeline
    pipeline = read(pipeline_path)
    if pipeline.get('status') != 'prepared_no_formal_training':
        raise ValueError('Pipeline is not complete')
    if not pipeline['stages'] or any(stage['exit_code'] for stage in pipeline['stages']):
        raise ValueError('A preparation step failed')
    smoke = read(ROOT / args.smoke)
    if not smoke['passed'] or smoke['skipped'] or smoke['source_snapshot_sha256'] != source_snapshot()[1]:
        raise ValueError('A current, fully passing synthetic smoke report is required')
    if read(ROOT / 'reports/environment_check.json')['status'] != 'ready':
        raise ValueError('Environment is not ready')
    prepared = read(pipeline['core_manifest'])
    for item in prepared['files'].values():
        if sha256_file(item['source_path']) != item['source_sha256']:
            raise ValueError('Raw data changed')
        if sha256_file(ROOT / 'data/processed/core' / item['parquet_path']) != item['parquet_sha256']:
            raise ValueError('Core cache changed')
    for path in pipeline['preflights']:
        preflight = read(path)
        if not preflight['inputs_ready'] or not preflight.get('frozen_sources_verified'):
            raise ValueError('Frozen source preflight failed')
    configs = list((ROOT / 'configs/experiments').glob('*.yaml'))
    verified_configs = {read(path)['config_sha256'] for path in pipeline['preflights']}
    for path in configs:
        execution = yaml.safe_load(path.read_text(encoding='utf-8'))['execution']
        if execution.get('stage') != 'preparation' or execution.get('allow_real_training') is not False:
            raise ValueError('A configuration is no longer in preparation mode')
        if sha256_file(path) not in verified_configs:
            raise ValueError('Experiment configuration changed after preflight')
    splits = read(pipeline['split_report'])
    if splits['status'] != 'complete' or not splits['holdout_isolation_checked']:
        raise ValueError('Frozen split is not isolated and complete')
    if sha256_file(ROOT / 'configs/splits.yaml') != splits['config_file_sha256']:
        raise ValueError('Split configuration changed')
    if sha256_file(ROOT / 'data/processed/core/train.parquet') != splits['train_parquet_sha256']:
        raise ValueError('Split core source changed')
    key_root = ROOT / 'artifacts/splits' / splits['split_id']
    if sha256_file(key_root / 'holdout_ids.parquet') != splits['holdout']['file_sha256']:
        raise ValueError('Holdout keys changed')
    for fold, detail in splits['folds'].items():
        for phase, item in detail['artifacts'].items():
            if sha256_file(key_root / fold / f'{phase}_ids.parquet') != item['file_sha256']:
                raise ValueError('Frozen cohort keys changed')
    if (list((ROOT / 'reports/runs').glob('*/run_manifest.json')) or
            list((ROOT / 'artifacts/runs').glob('**/*.pkl')) or
            list((ROOT / 'outputs/runs').glob('**/*.csv'))):
        raise ValueError('Run artifacts exist; inspect before claiming no formal training')
    development = read(ROOT / 'reports/audit_development.json')
    if (development['holdout_label_statistics_emitted'] is not False or
            development['holdout_key_sha256'] != splits['holdout']['file_sha256'] or
            development['preparation_manifest_sha256'] != sha256_file(pipeline['core_manifest'])):
        raise ValueError('Development audit provenance differs from frozen inputs')
    report = {'checked_at': timestamp(), 'status': 'prepared_no_formal_training',
              'formal_training_started': False, 'real_validation_scores_produced': False,
              'pipeline': str(pipeline_path), 'pipeline_sha256': sha256_file(pipeline_path),
              'pipeline_source_snapshot_sha256': pipeline['source_snapshot_sha256'],
              'current_source_snapshot_sha256': source_snapshot()[1],
              'synthetic_tests_passed': smoke['tests_run'],
              'preflight_count': len(pipeline['preflights']),
              'rows': {name: item['rows'] for name, item in prepared['files'].items()},
              'split_id': splits['split_id'], 'holdout_rows': splits['holdout']['n'],
              'remaining_target_readiness': development['remaining_target_readiness']}
    hashes, snapshot = source_snapshot()
    atomic_write_json(ROOT / 'reports/preparation/code_snapshot.json', {
        'created_at': timestamp(), 'files_sha256': hashes, 'snapshot_sha256': snapshot})
    atomic_write_json(ROOT / 'reports/preparation_status.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()

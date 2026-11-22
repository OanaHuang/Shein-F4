"""Preparation preflight or explicitly authorized, scoped real-data backtest."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from f4.common import atomic_write_json, project_path, sha256_file, timestamp
from f4.safety import require_preparation_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold', default='dev_1')
    parser.add_argument('--run-id')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute-training', action='store_true',
                      help='Requires an explicitly authorized training configuration')
    args = parser.parse_args()
    if not args.execute_training:
        require_preparation_mode(args.dry_run)
    path = project_path(args.config)
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if args.execute_training:
        if not args.run_id:
            parser.error('--execute-training requires an explicit --run-id')
        from f4.backtest import run_backtest
        result = run_backtest(config, args.fold, args.run_id, execute_training=True)
        from f4.platform_utils import current_process_peak_memory
        atomic_write_json(Path(result['locations']['reports']) / 'resources.json',
                          {'checked_at': timestamp(), **current_process_peak_memory()})
        print(json.dumps(result, ensure_ascii=False, default=str))
        return 0
    if config.get('execution', {}).get('allow_real_training') is not False:
        raise ValueError('Preparation config must explicitly set allow_real_training: false')
    split_path = project_path(config['splits'])
    splits = yaml.safe_load(split_path.read_text(encoding='utf-8'))
    split_id = splits['split_id']
    fold_names = {fold['name'] for fold in splits['development']}
    fold_names.add('holdout')
    if args.fold not in fold_names:
        raise ValueError(f'Unknown fold: {args.fold}')
    base = project_path(config.get('artifacts_base', 'artifacts')) / 'splits' / split_id
    data_dir = project_path(config.get('data_dir', 'data/processed/core'))
    split_report_path = project_path(config.get('split_reports_base',
        str(project_path(config.get('report_base', 'reports')) / 'splits'))) / f'{split_id}.json'
    required = [data_dir / 'train.parquet', data_dir / 'test.parquet',
                base / 'holdout_ids.parquet',
                split_report_path]
    from f4.models import make_conditional_model, QuantileBaseline
    from f4.targets import TargetTransform
    target = TargetTransform(config.get('target', 'total'))
    if config.get('experiment') not in {'B0', 'B1', 'M1', 'M2'}:
        raise ValueError('Unknown experiment')
    if config['experiment'] in {'B0', 'B1'} and (
            target.kind != 'total' or config.get('model_family', 'lightgbm') != 'lightgbm'):
        raise ValueError('B0/B1 support total-hour baseline configuration only')
    params = dict(config.get('model', {}))
    if config['experiment'] in {'M1', 'M2'}:
        if params.pop('alphas', [.9, .97]) != [.9, .97]:
            raise ValueError('M1/M2 require quantiles .90 and .97')
        make_conditional_model(config.get('model_family', 'lightgbm'), params)
    else:
        QuantileBaseline(kind='global' if config['experiment'] == 'B0' else 'carrier',
                         min_group_samples=params.get('min_group_samples', 200))
    report = {
        'model_family': config.get('model_family', 'lightgbm'), 'target': target.kind,
        'calibration_enabled': config['experiment'] == 'M2' or (
            config['experiment'] == 'B1' and config.get('calibration', {}).get('enabled') is True),
        'preflight_scope': 'source integrity and model parameters; not target viability or model performance',
        'created_at': timestamp(), 'stage': 'preparation', 'dry_run': True,
        'formal_training_started': False, 'validation_scored': False,
        'experiment': config['experiment'], 'fold': args.fold, 'split_id': split_id,
        'config_sha256': sha256_file(path), 'splits_config_sha256': sha256_file(split_path),
        'config_snapshot': config,
        'inputs': [{'path': str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                    'exists': p.exists()} for p in required],
    }
    report['inputs_ready'] = all(item['exists'] for item in report['inputs'])
    if report['inputs_ready']:
        frozen = json.loads(split_report_path.read_text(encoding='utf-8'))
        if (frozen.get('status') != 'complete' or not frozen.get('holdout_isolation_checked')
                or frozen['config_file_sha256'] != sha256_file(split_path)
                or frozen['train_parquet_sha256'] != sha256_file(data_dir / 'train.parquet')):
            raise ValueError('Frozen split/source integrity check failed')
        detail = frozen['folds'][args.fold]
        for phase, artifact in detail['artifacts'].items():
            if sha256_file(base / args.fold / f'{phase}_ids.parquet') != artifact['file_sha256']:
                raise ValueError(f'Frozen {phase} key hash changed')
        if sha256_file(base / 'holdout_ids.parquet') != frozen['holdout']['file_sha256']:
            raise ValueError('Frozen holdout key hash changed')
        report['frozen_sources_verified'] = True
        report['segments'] = {phase: values['n'] for phase, values in detail['segments'].items()}
        report['calibration_mature_fraction'] = detail['calibration_maturity']['mature_fraction']
        report['calibration_selection_bias_warning'] = (
            'As-of labels omit some slow parcels; maturity fraction does not prove unbiased calibration.')
    run_id = args.run_id or f"preflight_{config['experiment']}_{args.fold}"
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        raise ValueError('run_id must be a single safe path component')
    out = project_path('reports/preparation') / f'{run_id}.json'
    if out.exists():
        raise FileExistsError(f'Existing plan report must not be overwritten: {out}')
    atomic_write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, default=str))
    return 0 if report['inputs_ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

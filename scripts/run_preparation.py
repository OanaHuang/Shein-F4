"""Prepare local data and frozen splits; never invoke real model fitting/scoring."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import yaml
from f4.common import atomic_write_json, sha256_file, source_snapshot, timestamp
from f4.platform_utils import lock_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--run-id', required=True, help='Unique preparation ID; existing reports are not overwritten')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run-id')
    configurations = sorted((ROOT / 'configs/experiments').glob('*.yaml'))
    for path in configurations:
        execution = yaml.safe_load(path.read_text(encoding='utf-8'))['execution']
        if execution.get('stage') != 'preparation' or execution.get('allow_real_training') is not False:
            raise ValueError(f'Preparation config is unlocked: {path}')
    split_path = ROOT / 'configs/splits.yaml'
    splits = yaml.safe_load(split_path.read_text(encoding='utf-8'))
    report_dir = ROOT / 'reports/preparation' / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / 'pipeline.json'
    report = {'started_at': timestamp(), 'status': 'running', 'stages': [],
              'formal_training_started': False, 'real_validation_scores_produced': False,
              'source_snapshot_sha256': source_snapshot()[1], 'split_id': splits['split_id']}
    environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def run(name, script, *arguments):
        started = time.perf_counter()
        log = report_dir / f'{name}.log'
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        print(f'Preparing: {name}', flush=True)
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        report['stages'].append({'name': name, 'finished_at': timestamp(), 'command': command,
                                 'exit_code': result.returncode, 'log': str(log),
                                 'elapsed_seconds': round(time.perf_counter() - started, 3)})
        atomic_write_json(report_path, report)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {log}')

    atomic_write_json(report_path, report)
    try:
        with (ROOT / 'reports/preparation.lock').open('a') as lock:
            lock_file(lock)
            run('environment', 'check_environment.py')
            run('smoke', 'smoke_test.py', '--report', report_dir / 'smoke_results.json')
            run('archive', 'verify_archive.py', '--archive', args.archive, '--extract')
            core = ROOT / 'data/processed/core'
            manifest = core / 'preparation_manifest.json'
            if manifest.exists():
                cached = json.loads(manifest.read_text(encoding='utf-8'))
                if cached.get('status') != 'complete':
                    raise ValueError('Existing core cache is incomplete')
                for entry in cached['files'].values():
                    if (sha256_file(entry['source_path']) != entry['source_sha256'] or
                            sha256_file(core / entry['parquet_path']) != entry['parquet_sha256']):
                        raise ValueError('Existing cache provenance changed; refusing reuse')
                report['core_cache_reused'] = True
            else:
                run('core', 'prepare_data.py')
            run('structure', 'audit_data.py', '--mode', 'structure')
            run('splits', 'make_splits.py', '--config', split_path)
            run('development', 'audit_data.py', '--mode', 'development', '--exclude-ids',
                ROOT / 'artifacts/splits' / splits['split_id'] / 'holdout_ids.parquet')
            preflights = []
            for config in configurations:
                for fold in splits['development']:
                    name = f'{args.run_id}_{config.stem}_{fold["name"]}'
                    run(name, 'backtest.py', '--config', config, '--fold', fold['name'],
                        '--run-id', name, '--dry-run')
                    preflights.append(str(ROOT / 'reports/preparation' / f'{name}.json'))
            report.update(status='prepared_no_formal_training', finished_at=timestamp(),
                          preflights=preflights, core_manifest=str(manifest),
                          split_report=str(ROOT / 'reports/splits' / f'{splits["split_id"]}.json'))
            atomic_write_json(report_path, report)
            atomic_write_json(ROOT / 'reports/preparation_status.json', report)
            print(f'Preparation complete: {report_path}', flush=True)
    except BaseException as exc:
        report.update(status='failed', finished_at=timestamp(), error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(report_path, report)
        raise


if __name__ == '__main__':
    main()

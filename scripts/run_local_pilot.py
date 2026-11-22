"""Run explicitly authorized development comparisons serially with worker telemetry."""

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
from f4.common import atomic_write_json, sha256_file, source_snapshot, timestamp
from f4.platform_utils import lock_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--fold', choices=['dev_1', 'dev_2'], default='dev_1')
    parser.add_argument('--config-dir', default='configs/local_pilot')
    parser.add_argument('--experiments', nargs='+',
                        choices=['baseline', 'quantile', 'calibrated', 'baseline_calibrated', 'calibrated_shrinkage', 'catboost'],
                        default=['baseline', 'quantile', 'calibrated'])
    args = parser.parse_args()
    if not args.execute_training:
        parser.error('Explicit --execute-training is required')
    if os.name != 'nt':
        parser.error('This pilot resource monitor currently supports Windows only')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_id):
        parser.error('Invalid run ID')
    report_dir = ROOT / 'reports/local_pilot' / args.run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    report_path = report_dir / 'summary.json'
    files, snapshot = source_snapshot()
    atomic_write_json(report_dir / 'code_snapshot.json', files)
    report = {'started_at': timestamp(), 'status': 'running', 'fold': args.fold,
              'synthetic': False, 'final_holdout_evaluated': False,
              'source_snapshot_sha256': snapshot, 'runs': [],
              'resource_scope': 'End-to-end elapsed time; memory OS lifetime peaks reported inside actual Python worker after backtest; not whole-system RAM.'}
    atomic_write_json(report_path, report)
    environment = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    try:
        with (ROOT / 'reports/local_pilot.lock').open('a') as lock:
            lock_file(lock)
            for name in args.experiments:
                config_path = ROOT / args.config_dir / f'{name}.yaml'
                run_id = f'{args.run_id}_{name}'
                command = [sys.executable, '-u', str(ROOT / 'scripts/backtest.py'),
                           '--config', str(config_path), '--fold', args.fold,
                           '--run-id', run_id, '--execute-training']
                log = report_dir / f'{name}.log'
                entry = {'run_id': run_id, 'config': str(config_path),
                         'config_sha256': sha256_file(config_path), 'command': command,
                         'started_at': timestamp(), 'status': 'running', 'log': str(log),
                         'peak_working_set_bytes': None, 'peak_commit_bytes': None}
                report['runs'].append(entry)
                atomic_write_json(report_path, report)
                print(f'Starting {run_id}', flush=True)
                started = time.perf_counter()
                with log.open('w', encoding='utf-8') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=stream, stderr=subprocess.STDOUT)
                    try:
                        while True:
                            entry['elapsed_seconds'] = round(time.perf_counter() - started, 3)
                            atomic_write_json(report_path, report)
                            if child.poll() is not None:
                                break
                            time.sleep(1)
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                        child.wait()
                        raise
                entry.update(exit_code=child.returncode, finished_at=timestamp(),
                             elapsed_seconds=round(time.perf_counter() - started, 3))
                if child.returncode:
                    entry['status'] = 'failed'
                    raise RuntimeError(f'{run_id} failed; see {log}')
                manifest_path = ROOT / 'reports/runs' / run_id / 'run_manifest.json'
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                if manifest['status'] != 'complete' or manifest['synthetic'] or manifest['fold'] != args.fold:
                    raise ValueError('Unexpected backtest status or scope')
                resources_path = manifest_path.parent / 'resources.json'
                resources = json.loads(resources_path.read_text(encoding='utf-8'))
                entry.update(resources=resources, resources_sha256=sha256_file(resources_path),
                             peak_working_set_bytes=resources.get('peak_working_set_bytes'),
                             peak_commit_bytes=resources.get('peak_commit_bytes'))
                entry.update(status='complete', manifest=str(manifest_path),
                             manifest_sha256=sha256_file(manifest_path),
                             metrics=manifest['metrics'], cohort_n=manifest['cohort_n'])
                atomic_write_json(report_path, report)
                print(f"Completed {run_id}: {entry['elapsed_seconds']} s, memory status {resources['status']}", flush=True)
            if source_snapshot()[1] != snapshot:
                raise ValueError('Source changed while pilot was running')
            report.update(status='complete', finished_at=timestamp())
            atomic_write_json(report_path, report)
            print(f'Pilot complete: {report_path}', flush=True)
    except BaseException as exc:
        report.update(status='failed', finished_at=timestamp(), error=f'{type(exc).__name__}: {exc}')
        atomic_write_json(report_path, report)
        raise


if __name__ == '__main__':
    main()

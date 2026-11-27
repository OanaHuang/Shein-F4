"""Run an explicitly authorized development-only training queue, one process at a time."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_worker(argv):
    """Only stdlib is imported until a CLI worker's expected files are checked."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--execution-snapshot')
    args, _ = parser.parse_known_args(argv)
    if not args.worker:
        return
    if not args.execute_training:
        raise SystemExit('Explicit --execute-training is required before reading an execution snapshot')
    if not args.execution_snapshot:
        raise SystemExit('Worker requires --execution-snapshot before importing core modules')
    path = Path(args.execution_snapshot)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise SystemExit('Execution snapshot must remain inside the project')
    try:
        expected = json.loads(path.read_text(encoding='utf-8'))
        payload = {key: value for key, value in expected.items() if key != 'snapshot_sha256'}
        digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        if expected.get('schema_version') != 1 or digest(payload) != expected.get('snapshot_sha256'):
            raise ValueError('Execution snapshot digest is invalid')
        files = []
        for directory, pattern in [('src', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.yaml')]:
            files.extend((ROOT / directory).rglob(pattern))
        files.extend(ROOT / name for name in ['AGENTS.md', 'README.md', 'pyproject.toml',
                                             'requirements.txt', 'requirements.lock.txt', 'official/tools/score.py'])
        def file_hash(file):
            with file.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        hashes = {str(file.relative_to(ROOT)): file_hash(file) for file in files}
        extras = {}
        for name in expected.get('extra_files_sha256', {}):
            file = Path(name).resolve()
            if not file.is_relative_to(ROOT):
                raise ValueError('Snapshot config paths must remain inside the project')
            extras[str(file)] = file_hash(file)
        versions = {}
        for name in ['numpy', 'pandas', 'pyarrow', 'lightgbm', 'scikit-learn', 'PyYAML', 'xgboost', 'ngboost']:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        environment = {'python_version': sys.version, 'python_executable': str(Path(sys.executable).absolute()),
                       'python_prefix': str(Path(sys.prefix).absolute()), 'platform': platform.system(),
                       'machine': platform.machine(), 'packages': versions}
        observed = {'schema_version': 1, 'files_sha256': hashes,
                    'extra_files_sha256': extras, 'environment': environment}
        if digest(observed) != expected['snapshot_sha256']:
            raise ValueError('Execution source/configuration/environment changed before core import')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f'Execution snapshot bootstrap rejected: {error}') from error


if __name__ == '__main__':
    _bootstrap_worker(sys.argv[1:])

sys.path.insert(0, str(ROOT / 'src'))
import yaml
from f4.common import (atomic_write_json, sha256_file, source_snapshot, timestamp,
                       execution_snapshot, verify_execution_snapshot, ExecutionInvalidatedError)


class ResourceMonitorError(RuntimeError):
    pass


def record_run_failure(run_id, status, error, failure_kind, observed=None):
    """Invalidate all completion records without overwriting model/prediction bytes."""
    for filename in ['run_manifest.json', 'resources.json']:
        path = ROOT / 'reports/runs' / run_id / filename
        record = json.loads(path.read_text()) if path.exists() else {'run_id': run_id}
        record.update(status=status, finished_at=timestamp(), error=str(error), failure_kind=failure_kind)
        if observed is not None:
            record['source_at_finish'] = observed
        atomic_write_json(path, record)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('A batch/run ID must be a single safe path component')
    return value


def local_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Training paths must remain inside the project')
    return path


def validate_job(job, *, require_unused=True):
    run_id = safe_id(job['run_id'])
    if job.get('fold') not in {'dev_1', 'dev_2'}:
        raise ValueError('The local queue only permits dev_1/dev_2; holdout stays sealed')
    config_path = local_path(job['config'])
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    execution = config.get('execution', {})
    if (execution.get('stage') != 'training'
            or execution.get('allow_real_training') is not True
            or execution.get('parallel_models') is not False):
        raise ValueError('A queue job requires explicit serial real-training configuration')
    for threads in [execution.get('cpu_threads', 4), config.get('model', {}).get('n_jobs', 4)]:
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 4:
            raise ValueError('Local training permits at most four threads')
    for key, default in [('artifacts_base', 'artifacts'), ('report_base', 'reports'),
                         ('output_base', 'outputs')]:
        if local_path(config.get(key, default)) != (ROOT / default).resolve():
            raise ValueError('The local queue uses standard project artifact directories')
    if require_unused and any((ROOT / base / 'runs' / run_id).exists()
                              for base in ['artifacts', 'reports', 'outputs']):
        raise FileExistsError(f'Run already exists; refusing overwrite: {run_id}')
    return config_path, config


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def worker(job, resource_report, *, execute_training=False, config_sha256=None, execution_expected=None):
    if execute_training is not True:
        raise ValueError('Explicit --execute-training is required')
    config_path, config = validate_job(job)
    if config_sha256 and sha256_file(config_path) != config_sha256:
        raise ValueError('Job configuration changed after queue validation')
    destination = local_path(resource_report)
    expected = (ROOT / 'reports/runs' / job['run_id'] / 'resources.json').resolve()
    if destination != expected or destination.exists():
        raise ValueError('Resource report must be a new file in the current run directory')
    execution_expected = execution_expected or execution_snapshot(extra_paths=[config_path])
    source_at_start = verify_execution_snapshot(execution_expected)
    from f4.backtest import run_backtest
    started = time.monotonic()
    report = {'run_id': job['run_id'], 'started_at': timestamp(), 'status': 'running',
              'source_at_start': source_at_start}
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt('Training worker received termination request')

    signal.signal(signal.SIGTERM, terminate)
    try:
        result = run_backtest(config, job['fold'], job['run_id'], execute_training=True,
                              expected_execution_snapshot=execution_expected)
        report['source_at_finish'] = verify_execution_snapshot(execution_expected)
        report.update(status='complete', metrics=result['metrics'])
        return result
    except BaseException as error:
        invalidated = isinstance(error, ExecutionInvalidatedError)
        if not invalidated:
            try:
                report['source_at_finish'] = verify_execution_snapshot(execution_expected)
            except ExecutionInvalidatedError as drift:
                error, invalidated = drift, True
        if invalidated:
            report['source_at_finish'] = error.observed
            record_run_failure(job['run_id'], 'invalidated', error, 'execution_snapshot_changed', error.observed)
        report.update(status='invalidated' if invalidated else 'failed',
                      error=f'{type(error).__name__}: {error}')
        if invalidated:
            report['failure_kind'] = 'execution_snapshot_changed'
            raise error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        report.update(finished_at=timestamp(), elapsed_seconds=round(time.monotonic() - started, 3),
                      peak_RSS_bytes=peak_rss_bytes(), rss_source='resource.RUSAGE_SELF')
        report['peak_RSS_GiB'] = round(report['peak_RSS_bytes'] / 1024 ** 3, 4)
        atomic_write_json(destination, report)


def read_rss_bytes(pid):
    try:
        result = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True)
        if result.returncode or not result.stdout.strip():
            return None
        value = int(result.stdout.strip())
        return value * 1024 if value >= 0 else None
    except (OSError, ValueError):
        return None


def stop_worker(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_batch(plan_path, *, execute_training=False):
    if execute_training is not True:
        raise ValueError('Explicit --execute-training is required')
    plan_path = local_path(plan_path)
    plan = yaml.safe_load(plan_path.read_text(encoding='utf-8'))
    batch_id = safe_id(plan['batch_id'])
    limit = plan.get('max_rss_gib', 10)
    interval = plan.get('poll_seconds', 5)
    for value, low, high in [(limit, 0, 12), (interval, 0, 30)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low < value <= high:
            raise ValueError('Invalid memory limit or polling interval')
    jobs = plan.get('jobs', [])
    if not jobs or len({job['run_id'] for job in jobs}) != len(jobs):
        raise ValueError('A queue needs nonempty jobs with unique run IDs')
    for job in jobs:
        validate_job(job)
    directory = ROOT / 'reports/training' / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    state_path = directory / 'status.json'
    hashes, snapshot = source_snapshot()
    expected_execution = execution_snapshot(extra_paths=[plan_path, *[local_path(job['config']) for job in jobs]])
    execution_path = directory / 'execution_snapshot.json'
    atomic_write_json(execution_path, expected_execution)
    state = {'batch_id': batch_id, 'pid': os.getpid(), 'status': 'running',
             'started_at': timestamp(), 'current_run_id': None, 'max_rss_gib': limit,
             'plan_sha256': sha256_file(plan_path), 'source_snapshot_sha256': snapshot,
             'execution_snapshot_sha256': expected_execution['snapshot_sha256'],
             'execution_snapshot_file': str(execution_path.relative_to(ROOT)),
             'execution_snapshot_scope': 'source, official scorer, plan/job configs and runtime identity',
             'execution_snapshot_limitation': 'Start/end checks do not detect edits reverted between checks',
             'max_rss_failures': 3,
             'code_files_sha256': hashes, 'jobs': [dict(job, status='pending',
                config_sha256=sha256_file(local_path(job['config']))) for job in jobs]}
    atomic_write_json(state_path, state)
    process = None
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt('Training queue received termination request')

    signal.signal(signal.SIGTERM, terminate)
    try:
        for job in state['jobs']:
            verify_execution_snapshot(expected_execution)
            validate_job(job)
            run_id = job['run_id']
            resources = ROOT / 'reports/runs' / run_id / 'resources.json'
            command = [sys.executable, str(ROOT / 'scripts/train_local.py'), '--worker',
                       '--config', job['config'], '--fold', job['fold'], '--run-id', run_id,
                       '--resource-report', str(resources), '--config-sha256', job['config_sha256'],
                       '--execution-snapshot', str(execution_path),
                       '--execute-training']
            environment = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            log_path = directory / f'{run_id}.log'
            with log_path.open('x', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                state['current_run_id'] = run_id
                job.update(status='running', pid=process.pid, started_at=timestamp(),
                           log=str(log_path.relative_to(ROOT)), sampled_peak_RSS_bytes=0,
                           rss_consecutive_failures=0)
                atomic_write_json(state_path, state)
                print(f'{timestamp()} started {run_id} pid={process.pid}', flush=True)
                while process.poll() is None:
                    current = read_rss_bytes(process.pid)
                    state['heartbeat_at'] = timestamp()
                    if current is not None:
                        job.pop('monitor_warning', None)
                        job['rss_consecutive_failures'] = 0
                        job['sampled_peak_RSS_bytes'] = max(job['sampled_peak_RSS_bytes'], current)
                        job['current_RSS_bytes'] = current
                        if current > limit * 1024 ** 3:
                            atomic_write_json(state_path, state)
                            job['error'] = f'Worker RSS exceeded {limit} GiB'
                            stop_worker(process)
                            raise MemoryError(job['error'])
                    else:
                        # A missing final sample is normal if the worker just exited.
                        if process.poll() is not None:
                            break
                        job['rss_consecutive_failures'] += 1
                        job['monitor_warning'] = 'Worker RSS unavailable for this sample'
                        if job['rss_consecutive_failures'] >= 3:
                            atomic_write_json(state_path, state)
                            raise ResourceMonitorError('resource_monitor_failed: three consecutive RSS samples unavailable')
                    atomic_write_json(state_path, state)
                    time.sleep(interval)
                code = process.wait()
            job.update(exit_code=code, finished_at=timestamp())
            if resources.exists():
                job['resources'] = json.loads(resources.read_text())
            # Check even the last job, before accepting any successful worker output.
            job['source_at_finish'] = verify_execution_snapshot(expected_execution)
            if code:
                manifest_path = ROOT / 'reports/runs' / run_id / 'run_manifest.json'
                manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                if manifest.get('status') == 'invalidated':
                    raise ExecutionInvalidatedError('Worker invalidated its execution snapshot', manifest.get('source_at_finish'))
                raise RuntimeError(f'{run_id} exited with status {code}; see {log_path}')
            manifest_path = ROOT / 'reports/runs' / run_id / 'run_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('status') != 'complete' or manifest.get('synthetic') is not False:
                raise RuntimeError('Worker did not finish a real-data run successfully')
            job.update(status='complete', metrics=manifest['metrics'])
            state['current_run_id'] = None
            atomic_write_json(state_path, state)
            print(f'{timestamp()} completed {run_id}: {job["metrics"]}', flush=True)
            process = None
        state.update(status='complete', finished_at=timestamp())
        atomic_write_json(state_path, state)
        return state
    except BaseException as error:
        if process is not None:
            stop_worker(process)
        if not isinstance(error, ExecutionInvalidatedError):
            try:
                verify_execution_snapshot(expected_execution)
            except ExecutionInvalidatedError as drift:
                error = drift
        invalidated = isinstance(error, ExecutionInvalidatedError)
        failure_kind = ('execution_snapshot_changed' if invalidated else
                        'resource_monitor_failed' if isinstance(error, ResourceMonitorError) else 'worker_failed')
        for job in state['jobs']:
            if job['status'] == 'running':
                status = 'invalidated' if invalidated else 'failed'
                job.update(status=status, finished_at=timestamp(), error=f'{type(error).__name__}: {error}',
                           failure_kind=failure_kind)
                record_run_failure(job['run_id'], status, error, failure_kind,
                                   error.observed if invalidated else None)
                resources = ROOT / 'reports/runs' / job['run_id'] / 'resources.json'
                if resources.exists():
                    job['resources'] = json.loads(resources.read_text())
        state.update(status='invalidated' if invalidated else 'failed', finished_at=timestamp(),
                     error=f'{type(error).__name__}: {error}', failure_kind=failure_kind)
        atomic_write_json(state_path, state)
        if invalidated:
            raise error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--fold')
    parser.add_argument('--run-id')
    parser.add_argument('--resource-report')
    parser.add_argument('--config-sha256')
    parser.add_argument('--execution-snapshot')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute_training:
        parser.error('Explicit --execute-training is required before reading a training plan')
    if args.worker:
        if not all([args.config, args.fold, args.run_id, args.resource_report, args.execution_snapshot]):
            parser.error('Worker requires config, fold, run ID, resource report and execution snapshot')
        worker({'config': args.config, 'fold': args.fold, 'run_id': args.run_id},
               args.resource_report, execute_training=True, config_sha256=args.config_sha256,
               execution_expected=json.loads(local_path(args.execution_snapshot).read_text()))
    else:
        if not args.plan:
            parser.error('--plan is required')
        run_batch(args.plan, execute_training=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

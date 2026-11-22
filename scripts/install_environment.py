"""Install project dependencies in an independent OS background job."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from f4.platform_utils import file_digest, lock_file
import shutil

from monitor_environment import ROOT, save, timestamp, verify, wheel_progress


def wait_for(process, report, progress_path=None):
    try:
        while True:
            code = process.poll()
            report['wheel_progress'] = wheel_progress()
            if progress_path is not None:
                report['cached_wheel_downloaded_bytes'] = progress_path.stat().st_size if progress_path.exists() else 0
            save(report)
            if code is not None:
                return code
            time.sleep(10)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def resume_cached_wheel(report):
    manifest_path = ROOT / '.cache/wheels/partial_manifest.json'
    if not manifest_path.exists():
        return
    item = json.loads(manifest_path.read_text(encoding='utf-8'))
    partial = ROOT / item['partial_path']
    final = ROOT / item['final_path']
    candidate = final if final.exists() else partial
    if not final.exists():
        report.update(phase='resuming_cached_wheel', cached_wheel=item['filename'])
        save(report)
        process = subprocess.Popen([
            shutil.which('curl.exe' if os.name == 'nt' else 'curl') or 'curl', '--location', '--fail', '--silent', '--show-error',
            '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '60',
            '--retry', '3', '--retry-delay', '5', '--continue-at', '-',
            '--output', str(partial), item['source_url'],
        ])
        code = wait_for(process, report, partial)
        if code:
            raise RuntimeError(f'Cached wheel download failed: curl {code}')
    if candidate.stat().st_size != item['expected_bytes']:
        raise ValueError('Cached wheel size does not match official PyPI metadata')
    with candidate.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    if digest != item['sha256']:
        raise ValueError('Cached wheel SHA-256 does not match official PyPI metadata')
    with ZipFile(candidate) as wheel:
        if wheel.testzip() is not None:
            raise ValueError('Cached wheel CRC check failed')
    if candidate == partial:
        partial.rename(final)
    item.update(status='verified_complete', verified_at=timestamp())
    manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + '\n')


def install_cached_wheels(report):
    """Explicit file arguments and --no-index prevent remote same-version selection."""
    items = []
    preserved = ROOT / 'reports/wheel_cache_manifest.json'
    if preserved.exists():
        for item in json.loads(preserved.read_text(encoding='utf-8'))['files']:
            if item.get('status') == 'preserved_complete':
                items.append((ROOT / item['target'], item['bytes'], item['sha256']))
    resumed = ROOT / '.cache/wheels/partial_manifest.json'
    if resumed.exists():
        item = json.loads(resumed.read_text(encoding='utf-8'))
        if item.get('status') == 'verified_complete':
            items.append((ROOT / item['final_path'], item['expected_bytes'], item['sha256']))
    verified = []
    for path, expected_bytes, expected_sha in items:
        if not path.exists():
            continue
        if path.stat().st_size != expected_bytes:
            raise ValueError(f'Cached wheel size mismatch: {path.name}')
        with path.open('rb') as stream:
            actual_sha = file_digest(stream, 'sha256').hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f'Cached wheel SHA-256 mismatch: {path.name}')
        with ZipFile(path) as wheel:
            if wheel.testzip() is not None:
                raise ValueError(f'Cached wheel CRC check failed: {path.name}')
            if not any(name.endswith('.dist-info/WHEEL') for name in wheel.namelist()):
                raise ValueError(f'Not a wheel archive: {path.name}')
        verified.append(path)
    report['cached_wheels_verified'] = [path.name for path in verified]
    report['phase'] = 'installing_cached_wheels'
    save(report)
    if not verified:
        return
    print(f'{timestamp()} Installing {len(verified)} verified local wheels without network', flush=True)
    process = subprocess.Popen([
        sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
        '--disable-pip-version-check', *map(str, verified),
    ])
    report.update(pip_pid=process.pid, pip_process_running=True)
    code = wait_for(process, report)
    report.update(cached_wheel_install_exit_code=code, pip_process_running=False)
    if code:
        raise RuntimeError(f'Offline cached wheel installation failed: pip {code}')
    report['cached_packages_installed'] = {
        name: importlib.metadata.version(name)
        for name in ('numpy', 'pandas', 'pyarrow')
        if any(path.name.startswith(name + '-') for path in verified)
    }
    save(report)


def retry_pause(report, seconds):
    report.update(status='retry_wait', phase='waiting_to_retry', retry_in_seconds=seconds)
    print(f'{timestamp()} pip failed; retrying in {seconds}s. Completed local wheels remain installed.', flush=True)
    while seconds > 0:
        save(report)
        interval = min(seconds, 10)
        time.sleep(interval)
        seconds -= interval
        report['retry_in_seconds'] = seconds


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Run the installer with the project virtual-environment Python')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-only', action='store_true',
                        help='Verify and install complete local wheels without any network access')
    args = parser.parse_args()
    os.chdir(ROOT)
    (ROOT / '.cache').mkdir(exist_ok=True)
    with (ROOT / '.cache/environment_install.lock').open('a') as lock:
        lock_file(lock)
        report = {
            'started_at': timestamp(), 'status': 'installing', 'phase': 'preparing',
            'python_executable': sys.executable, 'venv': str(ROOT / '.venv'),
            'installation_scope': 'project virtual environment only',
            'requirements_source': 'requirements.txt',
            'manager': os.environ.get('F4_MANAGER', 'independent_session'),
            'worker_pid': os.getpid(),
            'monitor_log': 'reports/environment_install.log',
            'imports_verified': False, 'lightgbm_training_verified': False,
            'requirements_lock_written': False,
        }
        save(report)
        try:
            if not args.cache_only:
                resume_cached_wheel(report)
            install_cached_wheels(report)
            if args.cache_only:
                report.update(status='cached_packages_installed', phase='cache_only_finished')
                save(report)
                return 0
            attempt = 0
            maximum_attempts = int(os.environ.get('F4_INSTALL_MAX_ATTEMPTS', '0'))
            while True:
                attempt += 1
                report.update(status='installing', phase='pip_install', attempt=attempt,
                              retry_in_seconds=0)
                print(f'{timestamp()} Dependency installation attempt {attempt}', flush=True)
                process = subprocess.Popen([
                    sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--cache-dir', str(ROOT / '.cache/pip'),
                    '--timeout', '30', '--retries', '2', '--find-links',
                    str(ROOT / '.cache/wheels'), '-r', str(ROOT / 'requirements.txt'),
                ])
                report.update(pip_pid=process.pid, pip_process_running=True)
                code = wait_for(process, report)
                report.update(pip_process_running=False, last_pip_exit_code=code)
                if code == 0 or (maximum_attempts and attempt >= maximum_attempts):
                    break
                retry_pause(report, min(30 * 2 ** min(attempt - 1, 4), 300))
            report.update(phase='verifying', pip_exit_code=code, pip_process_running=False)
            verify(report)
            report['pip_exit_code'] = code
            report['pip_exit_code_note'] = 'Captured from child process in independent background session.'
            if code and report['status'] == 'ready':
                report['status'] = 'pip_failed'
            report['phase'] = 'finished'
            save(report)
            return 0 if report['status'] == 'ready' else 1
        except Exception as exc:
            report.update(status='installation_incomplete', error=f'{type(exc).__name__}: {exc}')
            save(report)
            raise


if __name__ == '__main__':
    raise SystemExit(main())

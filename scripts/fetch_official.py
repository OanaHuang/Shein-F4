"""Resume the official download, record progress, then verify and extract."""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'reports/data_manifest.json'
PARTIAL = ROOT / 'data/raw/F4.zip.part'
SOURCE_URL = 'https://imupd-public.sheincorp.cn/d/external-contents/F4.zip'
EXPECTED_BYTES = 590772903


def prepare_download_directories():
    (ROOT / 'reports/downloads').mkdir(parents=True, exist_ok=True)
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)


def initialize_manifest():
    """Create the initial manifest while holding the download lock."""
    if MANIFEST.exists():
        return
    data = {
        'project': 'SHEIN F4',
        'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        'archive': {
            'source_url': SOURCE_URL,
            'target': 'data/raw/F4.zip',
            'partial_target': 'data/raw/F4.zip.part',
            'status': 'not_started',
            'bytes': 0,
            'downloaded_bytes': 0,
            'expected_bytes': EXPECTED_BYTES,
            'sha256': None,
            'zip_verified': False,
            'extracted': False,
            'last_error': None,
        },
        'training_status': 'not_started',
    }
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(MANIFEST)


def record(**values):
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    data['updated_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    data['archive'].update(values)
    temporary = MANIFEST.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(MANIFEST)


def main():
    os.chdir(ROOT)
    prepare_download_directories()
    with (ROOT / 'reports/downloads/F4.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Another F4 downloader is already running')
        initialize_manifest()
        (ROOT / 'reports/downloads/F4.pid').write_text(str(os.getpid()) + '\n')
        archive = ROOT / 'data/raw/F4.zip'
        if not archive.exists():
            expected = json.loads(MANIFEST.read_text())['archive'].get('expected_bytes')
            attempt = 0
            failures = 0
            while True:
                start_bytes = PARTIAL.stat().st_size if PARTIAL.exists() else 0
                if expected and start_bytes == expected:
                    break
                if expected and start_bytes > expected:
                    record(status='download_failed', last_error='File exceeds expected archive size')
                    return 1
                attempt += 1
                command = [
                    '/usr/bin/curl', '--ipv4', '--http1.1', '--tlsv1.2', '--tls-max', '1.2',
                    '--location', '--fail', '--silent', '--show-error',
                    '--connect-timeout', '20', '--speed-limit', '1024', '--speed-time', '60',
                    '--continue-at', '-',
                    '--dump-header', 'reports/downloads/F4.headers',
                    '--output', str(PARTIAL),
                ]
                # Each new curl process recomputes its offset from the current file.
                # Do not use curl's internal --retry: it can rewind output to the
                # offset from the first request and discard newly received bytes.
                # Default DNS is resolved afresh; a manual override is retried only
                # on alternate attempts, so an old CDN address cannot pin every try.
                address = os.environ.get('F4_CDN_IP') if attempt % 2 else None
                if address:
                    command += ['--resolve', f'imupd-public.sheincorp.cn:443:{address}']
                command += [SOURCE_URL]
                record(status='downloading', downloader_pid=os.getpid(),
                       manager=os.environ.get('F4_MANAGER', 'independent_session'),
                       service_label=os.environ.get('F4_SERVICE_LABEL'), attempt=attempt,
                       downloaded_bytes=start_bytes, attempt_started_bytes=start_bytes,
                       current_ip_override=address, next_retry_at=None)
                print(f'{datetime.now().isoformat()} attempt={attempt} resume_at={start_bytes}', flush=True)
                process = subprocess.Popen(command, stderr=subprocess.PIPE, text=True)
                try:
                    while True:
                        code = process.poll()
                        size = PARTIAL.stat().st_size if PARTIAL.exists() else 0
                        update = {'downloaded_bytes': size}
                        if size > start_bytes:
                            update.update(last_error=None, curl_exit_code=None)
                        record(**update)
                        if code is not None:
                            break
                        time.sleep(10)
                    error = process.communicate()[1].strip()
                except BaseException:
                    process.terminate()
                    process.communicate()
                    record(status='interrupted')
                    raise
                if code == 0 and (not expected or size == expected):
                    break
                error = error or f'Incomplete archive: {size}/{expected} bytes'
                print(f'{datetime.now().isoformat()} curl_exit={code}: {error}', flush=True)
                if code in {3, 23, 26, 33, 60, 77}:
                    record(status='download_failed', curl_exit_code=code, last_error=error)
                    return code
                failures = 0 if size > start_bytes else failures + 1
                delay = min(300, 15 * 2 ** min(failures, 5))
                deadline = datetime.now(ZoneInfo('Asia/Shanghai')) + timedelta(seconds=delay)
                record(status='waiting_for_network', curl_exit_code=code, last_error=error,
                       next_retry_at=deadline.isoformat(timespec='seconds'), retry_delay_seconds=delay)
                for _ in range(delay // 5):
                    time.sleep(5)
                    record(status='waiting_for_network', downloaded_bytes=size)
            archive = PARTIAL
        record(status='verifying')
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/verify_archive.py'),
            '--archive', str(archive), '--extract',
        ])
        if result.returncode:
            record(status='verification_or_extraction_failed')
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

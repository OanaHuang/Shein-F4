"""Validate installed dependencies; the only fit is a 100-row synthetic smoke."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from monitor_environment import verify, timestamp


def main():
    path = ROOT / 'reports/environment_check.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update(phase='verifying', check_scope='imports, pip check, 100 synthetic rows/5 trees/1 thread',
                  formal_training_started=False, worker_pid=os.getpid(), pip_exit_code=0,
                  pip_exit_code_note='Packages installed; rerun validation after macOS libomp installation.')
    verify(report)
    report['phase'] = 'finished'
    report['runtime_dependency'] = {'name': 'libomp', 'path': '/opt/homebrew/opt/libomp',
                                    'checked_at': timestamp()}
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)
    print(report['status'])
    return 0 if report['status'] == 'ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())

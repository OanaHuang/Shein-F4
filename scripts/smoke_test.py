"""Run focused synthetic smoke tests; never fits or scores real F4 data."""

import argparse
import io
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from f4.common import atomic_write_json, timestamp, source_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='reports/smoke/smoke_results.json')
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    print(stream.getvalue(), end='')
    report = {
        'checked_at': timestamp(), 'data_kind': 'synthetic', 'formal_training_started': False,
        'passed': result.wasSuccessful() and not result.skipped,
        'tests_run': result.testsRun, 'duration_seconds': round(time.perf_counter() - start, 3),
        'failures': [(str(test), detail) for test, detail in result.failures],
        'errors': [(str(test), detail) for test, detail in result.errors],
        'skipped': [(str(test), reason) for test, reason in result.skipped],
        'scope': 'score boundaries, keys, preparation guard, synthetic data/splits/features/models',
        'source_snapshot_sha256': source_snapshot()[1],
    }
    path = Path(args.report)
    atomic_write_json(path if path.is_absolute() else ROOT / path, report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

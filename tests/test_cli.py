"""Synthetic file/CLI smoke; writes only within a temporary directory."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_synthetic_scoring_and_submission_validation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            truth = pd.DataFrame({'pkg_no': ['SMOKE_001', 'SMOKE_002'],
                                  'lgs_company_nm': ['A', 'B'], 'y_total_hours': [20., 50.]})
            pred = pd.DataFrame({'pkg_no': ['SMOKE_002', 'SMOKE_001'], 'pred_hours': [50., 20.]})
            truth.to_csv(root / 'truth.csv', index=False)
            pred.to_csv(root / 'pred.csv', index=False)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'),
                                     '--truth', str(root / 'truth.csv'), '--pred', str(root / 'pred.csv'),
                                     '--out', str(root / 'score'), '--synthetic'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / 'score/metrics.json').read_text())['metrics']['A'], 0)
            args = [sys.executable, str(ROOT / 'scripts/validate_submission.py'),
                    '--pred', str(root / 'pred.csv'), '--items', str(root / 'truth.csv'),
                    '--report', str(root / 'validation.json')]
            valid = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertTrue(json.loads((root / 'validation.json').read_text())['valid'])
            pred.loc[0, 'pred_hours'] = 0
            pred.to_csv(root / 'pred.csv', index=False)
            invalid = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertFalse(json.loads((root / 'validation.json').read_text())['valid'])


if __name__ == '__main__':
    unittest.main()

"""Calibration-only slope fitting and strict frozen-model reuse."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from f4.calibration import ShrunkResidualCalibrator
from f4.backtest import run_backtest
from test_backtest import synthetic_project


class ShrinkageTests(unittest.TestCase):
    def test_unhelpful_variation_collapses_and_useful_variation_is_retained(self):
        calibrator = ShrunkResidualCalibrator(min_group_samples=2)
        calibrator.fit([30., 20., 10., 10., 20., 30.], [10., 20., 30., 10., 20., 30.],
                       ['bad'] * 3 + ['good'] * 3, fit_ids=['SMOKE_FIT'],
                       cal_ids=[f'SMOKE_CAL_{i}' for i in range(6)])
        self.assertEqual(calibrator.group_slopes_, {'bad': 0., 'good': 1.})
        values, detail = calibrator.apply([1., 100., 10., 30.], ['bad', 'bad', 'good', 'good'], return_diagnostics=True)
        self.assertEqual(values[0], values[1])
        self.assertEqual(values[3] - values[2], 20.)
        self.assertEqual(detail.model_variation_slope.tolist(), [0., 0., 1., 1.])
        np.testing.assert_equal(calibrator.predict([1., 100.], ['bad', 'bad']), values[:2])
        self.assertTrue(np.isfinite(calibrator.apply([10.], ['unseen'])).all())

    def test_constant_base_and_independent_key_guards(self):
        calibrator = ShrunkResidualCalibrator(min_group_samples=2)
        with self.assertRaises(ValueError):
            calibrator.fit([10., 20.], [5., 5.], ['A', 'A'], fit_ids=['SMOKE_1'], cal_ids=['SMOKE_1', 'SMOKE_2'])
        calibrator.fit([10., 20.], [5., 5.], ['A', 'A'], fit_ids=['SMOKE_FIT'], cal_ids=['SMOKE_1', 'SMOKE_2'])
        self.assertEqual(calibrator.group_slopes_['A'], 0.)
        np.testing.assert_equal(calibrator.apply([1., 100.], ['A', 'A']), [20., 20.])

    def test_reuse_keeps_model_and_cohorts_fixed_and_rejects_parameter_change(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root, richer=True)
            config.update(experiment='M2', model={'n_estimators': 5, 'n_jobs': 1,
                                                 'min_child_samples': 1, 'num_leaves': 3},
                          calibration={'min_group_samples': 1})
            original = run_backtest(config, 'dev_1', 'SMOKE_BASE', synthetic=True)
            candidate = deepcopy(config)
            candidate['reuse_model_run'] = 'SMOKE_BASE'
            candidate['calibration']['method'] = 'shrunk_carrier_residual'
            result = run_backtest(candidate, 'dev_1', 'SMOKE_SHRUNK', synthetic=True)
            self.assertEqual(result['source'], original['source'])
            self.assertEqual(result['model_fit_ids_sha256'], original['model_fit_ids_sha256'])
            self.assertEqual(result['reused_model']['model_sha256'], original['artifacts']['model']['sha256'])
            self.assertEqual(result['calibration_ids_sha256'], original['calibration_ids_sha256'])
            self.assertEqual(result['metrics'], result['official_metrics'])
            candidate['model']['num_leaves'] = 4
            with self.assertRaisesRegex(ValueError, 'parameters differ'):
                run_backtest(candidate, 'dev_1', 'SMOKE_CHANGED', synthetic=True)
            manifest = json.loads((root/'reports/runs/SMOKE_CHANGED/run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')

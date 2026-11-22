"""Target reconstruction and model interchange use independent synthetic data."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import importlib.util
import json
import os
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from test_backtest import synthetic_project
from f4.backtest import run_backtest
from f4.models import CatBoostQuantileModel, load_model, make_conditional_model
from f4.targets import TargetTransform
from f4.platform_utils import lock_file
from f4.safety import TrainingNotAuthorizedError


class WorkflowTests(unittest.TestCase):
    def test_catboost_threads_bound_pool_training_and_prediction(self):
        model = CatBoostQuantileModel({'n_jobs': 4, 'n_estimators': 5})
        self.assertEqual(model._new_regressor(.9).get_params()['thread_count'], 4)
        matrix = pd.DataFrame({'weight': [1., 2.]})
        target = np.array([10., 20.])
        regressor = Mock()
        with patch('catboost.Pool') as pool:
            model._fit_regressor(regressor, matrix, target, [])
            self.assertEqual(pool.call_args.kwargs['thread_count'], 4)
            regressor.fit.assert_called_once_with(pool.return_value)
        model._predict_regressor(regressor, matrix)
        regressor.predict.assert_called_once_with(matrix, thread_count=4)

    def test_memory_is_measured_inside_actual_python_worker(self):
        if os.name != 'nt':
            return
        code = "import sys,json; sys.path.insert(0,'src'); from f4.platform_utils import current_process_peak_memory; data=bytearray(64*1024*1024); print(json.dumps(current_process_peak_memory()))"
        result = subprocess.run([sys.executable, '-c', code],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'available')
        self.assertGreater(report['peak_working_set_bytes'], 64 * 1024 * 1024)

    def test_remaining_roundtrip_and_anomalies(self):
        frame = pd.DataFrame({'pay_time': ['2026-08-01'], 'issue_time': ['2026-08-02'],
                              'y_total_hours': [30.]})
        target = TargetTransform('remaining')
        np.testing.assert_equal(target.transform(frame).to_numpy(), [6.])
        np.testing.assert_equal(target.restore([6.], frame), [30.])
        for value in ['bad', '2026-07-31']:
            with self.assertRaises(ValueError):
                target.offset(frame.assign(issue_time=value))
        with self.assertRaises(ValueError):
            target.transform(frame.assign(y_total_hours=20))
        with self.assertRaises(ValueError):
            target.restore([1., 2.], frame)

    def test_model_and_resource_guards(self):
        for params in [{'iterations': 500}, {'thread_count': 12}, {'task_type': 'GPU'},
                       {'n_jobs': 8}, {'n_estimators': True}, {'depth': 14}]:
            with self.assertRaises(ValueError):
                CatBoostQuantileModel(params)
        with self.assertRaises(ValueError):
            make_conditional_model('unknown', {})
        with self.assertRaises(TrainingNotAuthorizedError):
            run_backtest({'experiment': 'M2', 'model_family': 'catboost',
                          'execution': {'stage': 'preparation', 'allow_real_training': False}},
                         'dev_1', 'blocked', execute_training=True)

    def test_remaining_lightgbm_end_to_end(self):
        self.check_run('lightgbm', {'min_child_samples': 1, 'num_leaves': 3})

    @unittest.skipUnless(importlib.util.find_spec('catboost'), 'Optional CatBoost is not installed')
    def test_catboost_remaining_end_to_end_and_unknown_category(self):
        self.check_run('catboost', {'depth': 2})

    def check_run(self, family, params):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root, richer=True)
            config.update(experiment='M2', model_family=family, target='remaining',
                          model={'n_estimators': 5, 'n_jobs': 1, **params},
                          calibration={'min_group_samples': 1})
            result = run_backtest(config, 'dev_1', 'SMOKE_NEW', synthetic=True)
            self.assertEqual(result['metrics'], result['official_metrics'])
            self.assertEqual(result['target'], 'remaining')
            self.assertEqual(result['model_family'], family)
            encoder = load_model(root / 'artifacts/runs/SMOKE_NEW/encoder.pkl')
            model = load_model(root / 'artifacts/runs/SMOKE_NEW/model.pkl')
            raw = pd.read_parquet(root / 'data/train.parquet').iloc[:2].copy()
            raw['shpp_city_nm'] = 'UNSEEN_CITY'
            from f4.features import SHARED_COLUMNS
            matrix = encoder.transform(raw[list(SHARED_COLUMNS)])
            quantiles = model.predict_quantiles(matrix)
            self.assertTrue(np.isfinite(quantiles.to_numpy()).all())
            self.assertTrue((quantiles.q97 >= quantiles.q90).all())
            self.assertEqual(matrix.attrs['unknown_categories']['shpp_city_nm']['unknown'], 2)
            config['model']['n_estimators'] = 6
            with self.assertRaises(TrainingNotAuthorizedError):
                run_backtest(config, 'dev_1', 'SMOKE_TOO_BIG', synthetic=True)

    def test_exclusive_lock(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'lock'
            with path.open('a') as first:
                lock_file(first)
                with path.open('a') as second:
                    with self.assertRaises(BlockingIOError):
                        lock_file(second)
            with path.open('a') as third:
                lock_file(third)

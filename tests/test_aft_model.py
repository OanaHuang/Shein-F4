"""AFT adapter checks with a mocked booster. No model is trained."""

from pathlib import Path
from statistics import NormalDist
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from f4.aft_model import XGBoostAFTModel
from f4.features import pkg_ids_sha256


class XGBoostAFTModelTests(unittest.TestCase):
    def setUp(self):
        self.keys = ['0001', '0002']
        self.encoder_keys_sha = pkg_ids_sha256(self.keys[:1])
        self.matrix = pd.DataFrame({'weight': [3., 5.], 'cat__lgs_company_nm': [2, 1]}, index=[11, 24])
        self.matrix.attrs.update(pkg_ids_sha256=pkg_ids_sha256(self.keys),
                                 feature_fit_ids_sha256=self.encoder_keys_sha,
                                 feature_encoder_sha256='frozen-encoder')

    @patch('f4.aft_model._xgboost')
    def test_passes_aft_bounds_and_maps_log_margins_to_commitment(self, xgboost):
        xgb = xgboost.return_value
        booster = xgb.train.return_value
        booster.predict.return_value = np.log([20., 160.])
        model = XGBoostAFTModel({'n_jobs': 1, 'n_estimators': 2})
        model.fit(self.matrix, [30., 40.], [30., np.inf], pkg_ids=self.keys,
                  encoder_fit_ids_sha256=self.encoder_keys_sha)
        kwargs = xgb.QuantileDMatrix.call_args.kwargs
        np.testing.assert_array_equal(kwargs['label_lower_bound'], [30., 40.])
        np.testing.assert_array_equal(kwargs['label_upper_bound'], [30., np.inf])
        self.assertEqual(kwargs['feature_types'], ['q', 'c'])
        self.assertEqual(kwargs['nthread'], 1)
        self.assertEqual(xgb.train.call_args.args[0]['objective'], 'survival:aft')
        self.assertEqual(xgb.train.call_args.args[0]['device'], 'cpu')
        self.assertEqual(xgb.train.call_args.kwargs, {'num_boost_round': 2})
        self.assertEqual(model.training_ids_sha256_, pkg_ids_sha256(self.keys))
        self.assertEqual(model.feature_fit_ids_sha256_, self.encoder_keys_sha)

        base = np.array([20., 160.])
        q90 = base * np.exp(.5 * NormalDist().inv_cdf(.90))
        q97 = base * np.exp(.5 * NormalDist().inv_cdf(.97))
        np.testing.assert_allclose(model.predict(self.matrix), [q90[0], q97[1] - 24])
        booster.predict.assert_called_once_with(xgb.DMatrix.return_value, output_margin=True)
        wrong_encoder = self.matrix.copy()
        wrong_encoder.attrs['feature_encoder_sha256'] = 'another-encoder'
        with self.assertRaisesRegex(ValueError, 'encoder differs'):
            model.predict(wrong_encoder)

    @patch('f4.aft_model._xgboost')
    def test_rejects_invalid_bounds_and_encoder_before_loading_xgboost(self, xgboost):
        model = XGBoostAFTModel()
        for lower, upper in (([0., 40.], [30., np.inf]), ([30., 40.], [40., np.inf]),
                             ([30., 40.], [np.inf, np.inf]), ([30., 40.], [30., np.nan])):
            with self.subTest(lower=lower, upper=upper), self.assertRaisesRegex(ValueError, 'AFT'):
                model.fit(self.matrix, lower, upper, pkg_ids=self.keys,
                          encoder_fit_ids_sha256=self.encoder_keys_sha)
        with self.assertRaisesRegex(ValueError, 'encoder'):
            model.fit(self.matrix, [30., 40.], [30., np.inf], pkg_ids=self.keys,
                      encoder_fit_ids_sha256=pkg_ids_sha256(self.keys))
        xgboost.assert_not_called()

    def test_rejects_thread_or_round_aliases_and_invalid_scale(self):
        for config in ({'nthread': 8}, {'n_jobs': 5}, {'n_jobs': True}, {'num_boost_round': 15},
                       {'aft_loss_distribution_scale': 0}, {'aft_loss_distribution_scale': np.nan}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                XGBoostAFTModel(config)


if __name__ == '__main__':
    unittest.main()

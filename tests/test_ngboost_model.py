"""NGBoost adapter API checks. The estimator is mocked throughout."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from f4.features import pkg_ids_sha256
from f4.ngboost_model import NGBoostSurvivalModel


class NGBoostSurvivalTests(unittest.TestCase):
    def setUp(self):
        self.ids = ['0001', '0002', '0003', '0004']
        self.encoder_fit_ids = pkg_ids_sha256(self.ids[:2])
        self.matrix = pd.DataFrame({'weight': [3., np.nan, 8., 2.], 'cat__lgs_company_nm': [2, 3, 1, 0]},
                                   index=[11, 24, 26, 32])
        self.matrix.attrs.update(pkg_ids_sha256=pkg_ids_sha256(self.ids),
                                 feature_fit_ids_sha256=self.encoder_fit_ids,
                                 feature_encoder_sha256='frozen-encoder')
        self.lower = [30., 40., 45., 55.]
        self.upper = [30., 40., np.inf, np.inf]

    def fit(self, model, lower=None, upper=None, encoder_fit_ids=None):
        return model.fit(self.matrix, self.lower if lower is None else lower,
                         self.upper if upper is None else upper, pkg_ids=self.ids,
                         encoder_fit_ids_sha256=(self.encoder_fit_ids if encoder_fit_ids is None
                                                 else encoder_fit_ids))

    @patch('f4.ngboost_model.PREDICTION_BATCH_ROWS', 2)
    @patch('f4.ngboost_model._ngboost_components')
    def test_passes_exact_and_censored_events_and_keeps_batch_order(self, components):
        survival, lognormal = Mock(), object()
        components.return_value = survival, lognormal
        estimator = survival.return_value
        model = self.fit(NGBoostSurvivalModel({'n_estimators': 2}))

        options = survival.call_args.kwargs
        self.assertIs(options['Dist'], lognormal)
        self.assertEqual(options['n_estimators'], 2)
        self.assertEqual(options['minibatch_frac'], .25)
        self.assertIsNone(options['early_stopping_rounds'])
        self.assertEqual(options['Base'].min_samples_leaf, 200)
        at_fit = estimator.fit.call_args.args[0]
        self.assertEqual(at_fit.format, 'csr')
        np.testing.assert_array_equal(at_fit.toarray(), [[3, 1, 0], [0, 0, 1], [8, 0, 0], [2, 0, 0]])
        np.testing.assert_array_equal(estimator.fit.call_args.kwargs['T'], self.lower)
        np.testing.assert_array_equal(estimator.fit.call_args.kwargs['E'], [1, 1, 0, 0])
        self.assertEqual(model.training_ids_sha256_, pkg_ids_sha256(self.ids))
        self.assertEqual(model.feature_fit_ids_sha256_, self.encoder_fit_ids)

        first, second = Mock(), Mock()
        first.ppf.side_effect = [[80., 90.], [100., 140.]]
        second.ppf.side_effect = [[50., 85.], [90., 100.]]
        estimator.pred_dist.side_effect = [first, second]
        np.testing.assert_array_equal(model.predict(self.matrix), [80., 116., 66., 85.])
        np.testing.assert_array_equal(estimator.pred_dist.call_args_list[0].args[0].toarray(), at_fit[:2].toarray())
        np.testing.assert_array_equal(estimator.pred_dist.call_args_list[1].args[0].toarray(), at_fit[2:].toarray())

        wrong_encoder = self.matrix.copy()
        wrong_encoder.attrs['feature_encoder_sha256'] = 'changed-encoder'
        with self.assertRaisesRegex(ValueError, 'encoder differs'):
            model.predict(wrong_encoder)

    @patch('f4.ngboost_model._ngboost_components')
    def test_invalid_or_misaligned_labels_never_load_ngboost(self, components):
        model = NGBoostSurvivalModel()
        for lower, upper in (([0., 40., 45., 55.], self.upper),
                             (self.lower, [35., 40., np.inf, np.inf]),
                             (self.lower, [np.inf] * 4), (self.lower, [30., np.nan, np.inf, np.inf]),
                             (pd.Series(self.lower), self.upper)):
            with self.subTest(lower=lower, upper=upper), self.assertRaisesRegex(ValueError, 'NGBoost'):
                self.fit(model, lower, upper)
        with self.assertRaisesRegex(ValueError, 'encoder'):
            self.fit(model, upper=self.lower)
        components.assert_not_called()

    def test_rejects_resource_bypasses_and_invalid_parameters(self):
        for kwargs in ({'n_jobs': 2}, {'n_jobs': True}, {'nthread': 1}, {'n_estimators': 0},
                       {'minibatch_frac': 0.}, {'learning_rate': np.nan}, {'early_stopping_rounds': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NGBoostSurvivalModel(kwargs)


if __name__ == '__main__':
    unittest.main()

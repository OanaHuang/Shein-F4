"""XGBoost AFT on total delivery hours, with exact and right-censored labels."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from .features import CATEGORICAL_FEATURES, pkg_ids_sha256, validate_pkg_ids
from .models import _validate_matrix, finite_vector, positive_prediction


def _xgboost():
    try:
        import xgboost
    except (ImportError, OSError) as exc:
        raise RuntimeError('XGBoost cannot load; install the optional AFT dependency and check OpenMP') from exc
    return xgboost


class XGBoostAFTModel:
    """One normal AFT booster, with a fixed log-time scale and no early stopping."""

    def __init__(self, params=None):
        self.params = dict(learning_rate=.05, max_depth=6, reg_lambda=5., n_estimators=800,
                           n_jobs=4, random_state=2026, aft_loss_distribution_scale=.5)
        overrides = dict(params or {})
        unexpected = set(overrides) - self.params.keys()
        if unexpected:
            raise ValueError(f'Unsupported XGBoost AFT parameters: {sorted(unexpected)}')
        self.params.update(overrides)
        for name in ('max_depth', 'n_estimators', 'n_jobs', 'random_state'):
            number = self.params[name]
            minimum = 0 if name == 'random_state' else 1
            if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.params['n_jobs'] > 4:
            raise ValueError('Local XGBoost AFT requires 1 to 4 threads')
        for name in ('learning_rate', 'reg_lambda', 'aft_loss_distribution_scale'):
            number = self.params[name]
            if (isinstance(number, bool) or not isinstance(number, (int, float))
                    or not math.isfinite(number) or number < 0):
                raise ValueError(f'{name} must be finite and nonnegative')
        if not 0 < self.params['learning_rate'] <= 1 or self.params['aft_loss_distribution_scale'] == 0:
            raise ValueError('learning_rate must be in (0, 1] and AFT scale must be positive')

    def _matrix_options(self, matrix):
        # FeatureBuilder already assigns stable nonnegative codes, including codes
        # for unknown and missing values. Mark them as categories, not ordinals.
        feature_types = ['c' if name in CATEGORICAL_FEATURES else 'q' for name in matrix]
        return {'feature_types': feature_types, 'enable_categorical': True, 'nthread': self.params['n_jobs']}

    def fit(self, matrix, lower, upper, *, pkg_ids, encoder_fit_ids_sha256):
        _validate_matrix(matrix)
        for bound in (lower, upper):
            if isinstance(bound, pd.Series) and not bound.index.equals(matrix.index):
                raise ValueError('AFT label index does not match model matrix order')
        lower = finite_vector(lower, 'AFT lower bound', positive=True)
        upper = np.asarray(upper, dtype='float64')
        if lower.shape != (len(matrix),) or upper.shape != lower.shape:
            raise ValueError('AFT label bounds must match matrix rows')
        exact = upper == lower
        if not exact.any() or not (exact | np.isposinf(upper)).all():
            raise ValueError('AFT requires at least one exact label and only exact or right-censored bounds')

        fingerprint = pkg_ids_sha256(validate_pkg_ids(pkg_ids, len(matrix)))
        if matrix.attrs.get('pkg_ids_sha256') != fingerprint:
            raise ValueError('Feature rows are not bound to these fitting package IDs')
        if not encoder_fit_ids_sha256 or matrix.attrs.get('feature_fit_ids_sha256') != encoder_fit_ids_sha256:
            raise ValueError('AFT feature rows must use the shared exact-label cohort encoder')
        encoder_digest = matrix.attrs.get('feature_encoder_sha256')
        if not encoder_digest:
            raise ValueError('Model matrix must carry the frozen encoder digest')

        xgb = _xgboost()
        train = xgb.QuantileDMatrix(matrix, label_lower_bound=lower, label_upper_bound=upper,
                                    **self._matrix_options(matrix))
        scale = self.params['aft_loss_distribution_scale']
        options = {name: self.params[name] for name in ('learning_rate', 'max_depth', 'reg_lambda')}
        options.update(objective='survival:aft', eval_metric='aft-nloglik', tree_method='hist',
                       device='cpu', aft_loss_distribution='normal', aft_loss_distribution_scale=scale,
                       seed=self.params['random_state'], nthread=self.params['n_jobs'])
        self.booster_ = xgb.train(options, train, num_boost_round=self.params['n_estimators'])
        self.feature_names_ = list(matrix.columns)
        self.training_ids_sha256_ = fingerprint
        self.feature_fit_ids_sha256_ = encoder_fit_ids_sha256
        self.feature_encoder_sha256_ = encoder_digest
        self.training_rows_ = len(matrix)
        self.aft_loss_distribution_scale_ = scale
        return self

    def predict_quantiles(self, matrix):
        if not hasattr(self, 'booster_'):
            raise RuntimeError('XGBoostAFTModel has not been fitted')
        _validate_matrix(matrix)
        if list(matrix.columns) != self.feature_names_:
            raise ValueError('Prediction feature schema/order differs from fitting')
        if (matrix.attrs.get('feature_fit_ids_sha256') != self.feature_fit_ids_sha256_
                or matrix.attrs.get('feature_encoder_sha256') != self.feature_encoder_sha256_):
            raise ValueError('Prediction encoder differs from the fitted model encoder')

        xgb = _xgboost()
        data = xgb.DMatrix(matrix, **self._matrix_options(matrix))
        margin = finite_vector(self.booster_.predict(data, output_margin=True), 'AFT log-time score')
        if len(margin) != len(matrix):
            raise ValueError('AFT prediction length differs from model matrix')
        normal = NormalDist()
        quantiles = {}
        for name, probability in (('q90', .90), ('q97', .97)):
            with np.errstate(over='ignore', under='ignore'):
                hours = np.exp(margin + self.aft_loss_distribution_scale_ * normal.inv_cdf(probability))
            quantiles[name] = finite_vector(hours, name, positive=True)
        return pd.DataFrame(quantiles, index=matrix.index)

    def predict(self, matrix):
        quantiles = self.predict_quantiles(matrix)
        return positive_prediction(np.maximum(quantiles['q90'], quantiles['q97'] - 24))

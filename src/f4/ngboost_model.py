"""Log-normal NGBoost survival regression on exact and right-censored hours."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .features import CATEGORICAL_FEATURES, pkg_ids_sha256, validate_pkg_ids
from .models import _validate_matrix, finite_vector, positive_prediction


PREDICTION_BATCH_ROWS = 65_536


def _ngboost_components():
    try:
        from ngboost import NGBSurvival
        from ngboost.distns import LogNormal
    except (ImportError, OSError) as exc:
        raise RuntimeError('NGBoost cannot load; install the optional NGBoost dependency') from exc
    return NGBSurvival, LogNormal


def _tree_features(matrix, category_columns, encoder):
    # The shared feature matrix carries separate indicators for numeric
    # missingness/invalidity. Use finite zeros for sparse CART input.
    numbers = matrix.drop(columns=list(category_columns)).fillna(0).to_numpy(dtype=np.float32)
    numeric = csr_matrix(numbers)
    if encoder is None:
        return numeric
    categories = encoder.transform(matrix.loc[:, list(category_columns)])
    return hstack((numeric, categories), format='csr', dtype=np.float32)


class NGBoostSurvivalModel:
    """One distributional model that learns location and scale from the features."""

    def __init__(self, params=None):
        self.params = dict(learning_rate=.02, max_depth=3, min_samples_leaf=200,
                           n_estimators=250, minibatch_frac=.25, n_jobs=1, random_state=2026)
        overrides = dict(params or {})
        unexpected = set(overrides) - self.params.keys()
        if unexpected:
            raise ValueError(f'Unsupported NGBoost survival parameters: {sorted(unexpected)}')
        self.params.update(overrides)
        for name in ('max_depth', 'min_samples_leaf', 'n_estimators', 'n_jobs', 'random_state'):
            value = self.params[name]
            minimum = 0 if name == 'random_state' else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if self.params['n_jobs'] != 1:
            raise ValueError('Local NGBoost survival requires one thread')
        for name in ('learning_rate', 'minibatch_frac'):
            value = self.params[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f'{name} must be in (0, 1]')

    def fit(self, matrix, lower, upper, *, pkg_ids, encoder_fit_ids_sha256):
        _validate_matrix(matrix)
        for bound in (lower, upper):
            if isinstance(bound, pd.Series) and not bound.index.equals(matrix.index):
                raise ValueError('NGBoost label index does not match model matrix order')
        lower = finite_vector(lower, 'NGBoost lower bound', positive=True)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != (len(matrix),) or upper.shape != lower.shape:
            raise ValueError('NGBoost label bounds must match matrix rows')
        observed = upper == lower
        if not observed.any() or not (observed | np.isposinf(upper)).all():
            raise ValueError('NGBoost requires at least one exact label and only exact or right-censored bounds')

        ids = validate_pkg_ids(pkg_ids, len(matrix))
        fingerprint = pkg_ids_sha256(ids)
        if matrix.attrs.get('pkg_ids_sha256') != fingerprint:
            raise ValueError('Feature rows are not bound to these fitting package IDs')
        exact_digest = pkg_ids_sha256(key for key, exact in zip(ids, observed) if exact)
        if (encoder_fit_ids_sha256 != exact_digest
                or matrix.attrs.get('feature_fit_ids_sha256') != exact_digest):
            raise ValueError('NGBoost feature rows must use the shared exact-label cohort encoder')
        encoder_digest = matrix.attrs.get('feature_encoder_sha256')
        if not encoder_digest:
            raise ValueError('Model matrix must carry the frozen encoder digest')

        survival, distribution = _ngboost_components()
        category_columns = tuple(name for name in matrix if name in CATEGORICAL_FEATURES)
        encoder = None
        if category_columns:
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
            encoder.fit(matrix.loc[observed, list(category_columns)])
        tree = DecisionTreeRegressor(max_depth=self.params['max_depth'],
                                     min_samples_leaf=self.params['min_samples_leaf'],
                                     random_state=self.params['random_state'])
        model = survival(Dist=distribution, Base=tree, n_estimators=self.params['n_estimators'],
                         learning_rate=self.params['learning_rate'], minibatch_frac=self.params['minibatch_frac'],
                         random_state=self.params['random_state'], verbose=False, early_stopping_rounds=None)
        with threadpool_limits(limits=self.params['n_jobs']):
            model.fit(_tree_features(matrix, category_columns, encoder), T=lower, E=observed.astype(np.uint8))

        self.model_ = model
        self.category_columns_ = category_columns
        self.category_encoder_ = encoder
        self.feature_names_ = list(matrix.columns)
        self.training_ids_sha256_ = fingerprint
        self.feature_fit_ids_sha256_ = exact_digest
        self.feature_encoder_sha256_ = encoder_digest
        self.training_rows_ = len(matrix)
        return self

    def predict_quantiles(self, matrix):
        if not hasattr(self, 'model_'):
            raise RuntimeError('NGBoostSurvivalModel has not been fitted')
        _validate_matrix(matrix)
        if list(matrix.columns) != self.feature_names_:
            raise ValueError('Prediction feature schema/order differs from fitting')
        if (matrix.attrs.get('feature_fit_ids_sha256') != self.feature_fit_ids_sha256_
                or matrix.attrs.get('feature_encoder_sha256') != self.feature_encoder_sha256_):
            raise ValueError('Prediction encoder differs from the fitted model encoder')

        q90 = np.empty(len(matrix))
        q97 = np.empty(len(matrix))
        with threadpool_limits(limits=self.params['n_jobs']):
            for start in range(0, len(matrix), PREDICTION_BATCH_ROWS):
                rows = slice(start, start + PREDICTION_BATCH_ROWS)
                features = _tree_features(matrix.iloc[rows], self.category_columns_, self.category_encoder_)
                distribution = self.model_.pred_dist(features)
                for probability, output in ((.90, q90), (.97, q97)):
                    values = finite_vector(distribution.ppf(probability), f'NGBoost Q{probability:.0%}', positive=True)
                    if values.shape != output[rows].shape:
                        raise ValueError('NGBoost quantile length differs from the prediction batch')
                    output[rows] = values
        if (q97 < q90).any():
            raise ValueError('NGBoost Q97 is below Q90')
        return pd.DataFrame({'q90': q90, 'q97': q97}, index=matrix.index)

    def predict(self, matrix):
        quantiles = self.predict_quantiles(matrix)
        return positive_prediction(np.maximum(quantiles['q90'], quantiles['q97'] - 24))

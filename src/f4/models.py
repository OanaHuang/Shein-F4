"""B0/B1 and sequential LightGBM conditional quantiles.

No CLI invokes these fit methods on real data during preparation. Training
authorization belongs to the guarded project entry point.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .features import (CATEGORICAL_FEATURES, FEATURE_COLUMNS, pkg_ids_sha256,
                       validate_pkg_ids)

ITERATION_ALIASES = frozenset({
    'max_iter', 'n_iter', 'nrounds', 'num_boost_round', 'num_iteration',
    'num_iterations', 'num_round', 'num_rounds', 'num_tree', 'num_trees',
})


def finite_vector(values, name='values', positive=False) -> np.ndarray:
    result = np.asarray(values, dtype='float64')
    if result.ndim != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a nonempty finite vector')
    if positive and not (result > 0).all():
        raise ValueError(f'{name} must be strictly positive')
    return result


def empirical_quantile(values, q: float) -> float:
    """The ceil(q*n)-th order statistic, without interpolation."""
    data = finite_vector(values)
    if not 0 < q <= 1:
        raise ValueError('q must be in (0, 1]')
    position = math.ceil(q * len(data)) - 1
    return float(np.partition(data, position)[position])


def commitment_constant(values) -> float:
    return max(empirical_quantile(values, 0.90), empirical_quantile(values, 0.97) - 24.0)


def group_values(values, n: int) -> pd.Series:
    result = pd.Series(values, dtype='string').reset_index(drop=True)
    if len(result) != n:
        raise ValueError('group length does not match rows')
    return result.replace('', pd.NA)


def positive_prediction(values, minimum: float = 1e-6) -> np.ndarray:
    result = finite_vector(values, 'predictions')
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError('minimum prediction must be finite and positive')
    return np.maximum(result, minimum)


class QuantileBaseline:
    """Global B0 or carrier B1; sparse/missing/new carriers use the global pool."""

    def __init__(self, kind='global', min_group_samples=200, min_prediction=1e-6):
        if kind not in {'global', 'carrier'}:
            raise ValueError('kind must be global or carrier')
        if min_group_samples < 1:
            raise ValueError('min_group_samples must be >= 1')
        self.kind = kind
        self.min_group_samples = int(min_group_samples)
        self.min_prediction = min_prediction

    def fit(self, y, groups=None, *, pkg_ids):
        target = finite_vector(y, 'y', positive=True)
        ids = validate_pkg_ids(pkg_ids, len(target))
        self.global_prediction_ = commitment_constant(target)
        self.training_rows_ = len(target)
        self.training_ids_sha256_ = pkg_ids_sha256(ids)
        self.group_counts_ = {}
        self.group_predictions_ = {}
        if self.kind == 'carrier' and groups is None:
            raise ValueError('Carrier baseline requires groups')
        if groups is not None:
            grouped = group_values(groups, len(target))
            frame = pd.DataFrame({'group': grouped, 'y': target})
            for group, rows in frame.groupby('group', observed=True):
                self.group_counts_[str(group)] = len(rows)
                if self.kind == 'carrier' and len(rows) >= self.min_group_samples:
                    self.group_predictions_[str(group)] = commitment_constant(rows['y'])
        return self

    def predict(self, groups=None, *, n_rows=None, return_diagnostics=False):
        if not hasattr(self, 'global_prediction_'):
            raise RuntimeError('Baseline has not been fitted')
        if groups is None:
            if n_rows is None or n_rows < 1:
                raise ValueError('Supply groups or a positive n_rows')
            if self.kind == 'carrier':
                raise ValueError('Carrier baseline requires groups at prediction')
            group = pd.Series([pd.NA] * int(n_rows), dtype='string')
        else:
            group = group_values(groups, len(groups))
            if n_rows is not None and len(group) != n_rows:
                raise ValueError('n_rows and groups disagree')
        mapped = group.map(self.group_predictions_) if self.kind == 'carrier' else pd.Series(np.nan, index=group.index)
        predictions = positive_prediction(mapped.fillna(self.global_prediction_).to_numpy(), self.min_prediction)
        if not return_diagnostics:
            return predictions
        diagnostics = pd.DataFrame({
            'group': group,
            'fitting_n': group.map(self.group_counts_).fillna(0).astype('int64'),
            'fallback_level': np.where(mapped.notna(), 'carrier', 'global'),
        })
        return predictions, diagnostics


def _validate_matrix(matrix: pd.DataFrame):
    if not isinstance(matrix, pd.DataFrame) or matrix.empty or matrix.columns.duplicated().any():
        raise ValueError('Model matrix must be a nonempty DataFrame with unique columns')
    extra = set(matrix.columns) - set(FEATURE_COLUMNS)
    if extra:
        raise ValueError(f'Unsupported or forbidden model features: {sorted(extra)}')
    for column in matrix:
        if not is_numeric_dtype(matrix[column]) or np.isinf(matrix[column].to_numpy()).any():
            raise ValueError(f'Model feature must be numeric without infinity: {column}')


class ConditionalQuantileModel:
    """M1: two models trained serially, with no evaluation labels/early stopping."""

    def __init__(self, params=None, min_prediction=1e-6):
        self.params = dict(learning_rate=0.05, num_leaves=31, min_child_samples=200,
                           reg_lambda=5.0, n_estimators=800, random_state=2026,
                           n_jobs=4, force_col_wise=True, verbosity=-1)
        overrides = dict(params or {})
        if set(overrides).intersection({'loss', 'application', 'objective_type', 'app'}):
            raise ValueError('Objective aliases are not allowed; use objective=quantile only')
        if set(overrides).intersection(ITERATION_ALIASES):
            raise ValueError('Use n_estimators only; iteration aliases are not allowed')
        if overrides.pop('objective', 'quantile') != 'quantile' or 'alpha' in overrides:
            raise ValueError('This model fixes objective=quantile and alpha=0.90/0.97')
        self.params.update(overrides)
        if (isinstance(self.params['n_estimators'], bool)
                or not isinstance(self.params['n_estimators'], int)
                or self.params['n_estimators'] < 1):
            raise ValueError('n_estimators must be a positive integer')
        if set(overrides).intersection({'num_threads', 'nthread', 'num_thread', 'nthreads'}):
            raise ValueError('Use n_jobs for the explicit 1-to-4-thread resource limit')
        if not 1 <= self.params['n_jobs'] <= 4:
            raise ValueError('Local project training requires 1 to 4 threads')
        self.min_prediction = min_prediction

    def fit(self, matrix, y, *, pkg_ids, categorical_features=None):
        _validate_matrix(matrix)
        target = finite_vector(y, 'y', positive=True)
        if len(target) != len(matrix):
            raise ValueError('y length does not match matrix')
        if isinstance(y, pd.Series) and not y.index.equals(matrix.index):
            raise ValueError('y index does not match model matrix order')
        ids = validate_pkg_ids(pkg_ids, len(matrix))
        fingerprint = pkg_ids_sha256(ids)
        if matrix.attrs.get('pkg_ids_sha256') != fingerprint:
            raise ValueError('Feature rows are not bound to these fitting package IDs')
        if matrix.attrs.get('feature_fit_ids_sha256') != fingerprint:
            raise ValueError('Category dictionaries must be fitted on this fitting cohort only')
        encoder_digest = matrix.attrs.get('feature_encoder_sha256')
        if not encoder_digest:
            raise ValueError('Model matrix must carry the frozen encoder digest')
        categories = list(categorical_features if categorical_features is not None
                          else [c for c in CATEGORICAL_FEATURES if c in matrix])
        if set(categories) - set(CATEGORICAL_FEATURES) or set(categories) - set(matrix):
            raise ValueError('Invalid categorical feature list')
        try:
            from lightgbm import LGBMRegressor
        except (ImportError, OSError) as exc:
            raise RuntimeError('LightGBM cannot load; check the project environment and macOS libomp') from exc
        models = {}
        for alpha in (0.90, 0.97):
            model = LGBMRegressor(objective='quantile', alpha=alpha, **self.params)
            model.fit(matrix, target, categorical_feature=categories)
            models[alpha] = model
        self.models_ = models
        self.feature_names_ = list(matrix.columns)
        self.training_ids_sha256_ = fingerprint
        self.feature_encoder_sha256_ = encoder_digest
        self.training_rows_ = len(matrix)
        return self

    def predict_quantiles(self, matrix):
        if not hasattr(self, 'models_'):
            raise RuntimeError('ConditionalQuantileModel has not been fitted')
        _validate_matrix(matrix)
        if list(matrix.columns) != self.feature_names_:
            raise ValueError('Prediction feature schema/order differs from fitting')
        if matrix.attrs.get('feature_fit_ids_sha256') != self.training_ids_sha256_:
            raise ValueError('Prediction encoder is not the fitted model encoder')
        if matrix.attrs.get('feature_encoder_sha256') != self.feature_encoder_sha256_:
            raise ValueError('Prediction encoder dictionary differs from the fitted encoder')
        q90 = finite_vector(self.models_[0.90].predict(matrix), 'q90')
        q97 = np.maximum(finite_vector(self.models_[0.97].predict(matrix), 'q97'), q90)
        return pd.DataFrame({'q90': q90, 'q97': q97}, index=matrix.index)

    def predict(self, matrix):
        quantiles = self.predict_quantiles(matrix)
        return positive_prediction(np.maximum(quantiles['q90'], quantiles['q97'] - 24), self.min_prediction)


def save_model(model, path):
    """Only load these pickle artifacts from trusted local project runs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):
    """Load a trusted local artifact, never an arbitrary downloaded pickle."""
    with Path(path).open('rb') as stream:
        return pickle.load(stream)

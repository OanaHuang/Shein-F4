"""Independent carrier residual calibration for the route-A protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import pkg_ids_sha256, validate_pkg_ids
from .models import commitment_constant, finite_vector, group_values, positive_prediction


class ResidualCalibrator:
    """Fit offsets from independent predictions, with an auditable ID boundary.

    The caller must provide actual fitting/calibration keys. This check rejects
    overlap; it does not by itself prove that supplied predictions came from the
    claimed model. Bind ``model_fit_ids_sha256`` to that model's recorded digest.
    Empirical coverage on calibration data is not a future coverage guarantee.
    """

    def __init__(self, min_group_samples=200, min_prediction=1e-6):
        if min_group_samples < 1:
            raise ValueError('min_group_samples must be >= 1')
        self.min_group_samples = int(min_group_samples)
        self.min_prediction = min_prediction

    def fit(self, y, base_predictions, groups, *, fit_ids, cal_ids,
            model_fit_ids_sha256=None):
        target = finite_vector(y, 'y', positive=True)
        base = finite_vector(base_predictions, 'base_predictions', positive=True)
        if len(base) != len(target):
            raise ValueError('Calibration prediction length mismatch')
        fitting = validate_pkg_ids(fit_ids, name='fit_ids')
        calibration = validate_pkg_ids(cal_ids, len(target), 'cal_ids')
        if set(fitting).intersection(calibration):
            raise ValueError('Fitting and calibration package IDs must not overlap')
        fingerprint = pkg_ids_sha256(fitting)
        if model_fit_ids_sha256 is not None and model_fit_ids_sha256 != fingerprint:
            raise ValueError('Calibration fitting IDs do not match the base model digest')
        residual = target - base
        # commitment_constant works for residuals too: it does not impose
        # positivity. A negative offset is intentional and valid.
        self.global_delta_ = commitment_constant(residual)
        self.fit_ids_sha256_ = fingerprint
        self.cal_ids_sha256_ = pkg_ids_sha256(calibration)
        self.calibration_rows_ = len(target)
        self.group_counts_ = {}
        self.group_deltas_ = {}
        frame = pd.DataFrame({'group': group_values(groups, len(target)), 'residual': residual})
        for group, rows in frame.groupby('group', observed=True):
            self.group_counts_[str(group)] = len(rows)
            if len(rows) >= self.min_group_samples:
                self.group_deltas_[str(group)] = commitment_constant(rows['residual'])
        return self

    def apply(self, base_predictions, groups, *, return_diagnostics=False):
        if not hasattr(self, 'global_delta_'):
            raise RuntimeError('ResidualCalibrator has not been fitted')
        base = finite_vector(base_predictions, 'base_predictions', positive=True)
        grouped = group_values(groups, len(base))
        mapped = grouped.map(self.group_deltas_)
        delta = mapped.fillna(self.global_delta_).to_numpy()
        prediction = positive_prediction(base + delta, self.min_prediction)
        if not return_diagnostics:
            return prediction
        diagnostics = pd.DataFrame({
            'group': grouped,
            'calibration_n': grouped.map(self.group_counts_).fillna(0).astype('int64'),
            'fallback_level': np.where(mapped.notna(), 'carrier', 'global'),
            'delta': delta,
        })
        return prediction, diagnostics

    predict = apply


class ShrunkResidualCalibrator(ResidualCalibrator):
    """Attenuate unreliable within-carrier variation using calibration labels only.

    Each slope is cov(base, y) / var(base), clipped to [0, 1]. This is a fixed
    fitting rule, not a search on evaluation scores. Independent residual
    quantiles then set the commitment level. No future coverage is guaranteed.
    """

    @staticmethod
    def _estimate(base, target):
        center = float(np.mean(base))
        centered = base - center
        variance = float(np.mean(centered * centered))
        slope = 0.0 if variance == 0 else float(np.clip(
            np.mean(centered * (target - np.mean(target))) / variance, 0., 1.))
        return center, slope

    def _transform_base(self, base, groups):
        grouped = group_values(groups, len(base))
        centers = grouped.map(self.group_centers_).fillna(self.global_center_).to_numpy(dtype=float)
        slopes = grouped.map(self.group_slopes_).fillna(self.global_slope_).to_numpy(dtype=float)
        return (1. - slopes) * centers + slopes * base, slopes

    def fit(self, y, base_predictions, groups, *, fit_ids, cal_ids,
            model_fit_ids_sha256=None):
        # Reuse all existing disjoint-key, digest, length and finite-value guards.
        super().fit(y, base_predictions, groups, fit_ids=fit_ids, cal_ids=cal_ids,
                    model_fit_ids_sha256=model_fit_ids_sha256)
        target = finite_vector(y, 'y', positive=True)
        base = finite_vector(base_predictions, 'base_predictions', positive=True)
        self.global_center_, self.global_slope_ = self._estimate(base, target)
        self.group_centers_, self.group_slopes_ = {}, {}
        frame = pd.DataFrame({'group': group_values(groups, len(base)), 'base': base, 'target': target})
        for group, rows in frame.groupby('group', observed=True):
            if len(rows) >= self.min_group_samples:
                center, slope = self._estimate(rows.base.to_numpy(), rows.target.to_numpy())
                self.group_centers_[str(group)] = center
                self.group_slopes_[str(group)] = slope
        adjusted, _ = self._transform_base(base, groups)
        return super().fit(y, adjusted, groups, fit_ids=fit_ids, cal_ids=cal_ids,
                           model_fit_ids_sha256=model_fit_ids_sha256)

    def apply(self, base_predictions, groups, *, return_diagnostics=False):
        if not hasattr(self, 'global_slope_'):
            raise RuntimeError('ShrunkResidualCalibrator has not been fitted')
        base = finite_vector(base_predictions, 'base_predictions', positive=True)
        adjusted, slopes = self._transform_base(base, groups)
        result = super().apply(adjusted, groups, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            prediction, diagnostics = result
            diagnostics['model_variation_slope'] = slopes
            return prediction, diagnostics
        return result

    predict = apply

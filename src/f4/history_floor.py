"""Fixed carrier tail floor from sufficiently old, already-signed history."""

import numpy as np
import pandas as pd

from .features import pkg_ids_sha256, validate_pkg_ids
from .models import commitment_constant, finite_vector, group_values


def fit_history_floor(history, cutoff, min_age_days=21, min_group_samples=200):
    if min_age_days <= 0 or min_group_samples < 1:
        raise ValueError('History age and group size must be positive')
    if history.empty:
        raise ValueError('No eligible mature history; do not change the cutoff')
    validate_pkg_ids(history.pkg_no, len(history))
    cutoff = pd.Timestamp(cutoff)
    issue = pd.to_datetime(history.issue_time, errors='coerce')
    sign = pd.to_datetime(history.sign_time, errors='coerce')
    if (issue.isna().any() or sign.isna().any()
            or not issue.lt(cutoff - pd.Timedelta(days=min_age_days)).all()
            or not sign.lt(cutoff).all()):
        raise ValueError('History violates the fixed age or available-label cutoff')
    finite_vector(history.y_total_hours, 'history labels', positive=True)
    if history.lgs_company_nm.isna().any():
        raise ValueError('Missing historical carrier')
    groups = {}
    for carrier, rows in history.groupby('lgs_company_nm', observed=True):
        if len(rows) >= min_group_samples:
            groups[str(carrier)] = {'n': len(rows), 'floor_hours': commitment_constant(rows.y_total_hours)}
    return {'method': 'mature_history_carrier_floor_v1', 'cutoff': cutoff.isoformat(),
            'min_age_days': min_age_days, 'min_group_samples': min_group_samples,
            'history_n': len(history), 'history_keys_sha256': pkg_ids_sha256(history.pkg_no),
            'groups': groups, 'fallback': 'Keep the original prediction for sparse/unknown carriers',
            'scope': 'Historical target statistic, not independent residual calibration; history may overlap model fitting keys'}


def apply_history_floor(predictions, carriers, fitted):
    base = finite_vector(predictions, 'base predictions', positive=True)
    grouped = group_values(carriers, len(base))
    floors = grouped.map({k: v['floor_hours'] for k, v in fitted['groups'].items()}).to_numpy(dtype=float)
    return np.where(np.isfinite(floors), np.maximum(base, floors), base)

"""Build the development-only, right-censored supplement for AFT fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from .features import SHARED_COLUMNS
from .splits import _timestamp


def _package_keys(values, name):
    keys = pd.Index(values)
    if keys.hasnans or not is_string_dtype(keys) or (keys == '').any() or keys.has_duplicates:
        raise ValueError(f'{name} must contain unique nonempty string keys')
    return keys.astype('string')


def read_censored_fit(train_path, *, fit_ids, cal_ids, eval_ids, holdout_ids,
                      issue_cutoff, label_cutoff):
    """Read shared features for pre-F parcels unsigned at the strict C cutoff."""
    issue_cutoff = _timestamp(issue_cutoff, 'AFT fit issue cutoff')
    label_cutoff = _timestamp(label_cutoff, 'AFT fit label cutoff')
    if issue_cutoff >= label_cutoff:
        raise ValueError('AFT requires the issue cutoff before the label cutoff')

    issue_filter = ('issue_time', '<', issue_cutoff)
    # Establish the fitting candidate set from shared information before using
    # a signature predicate. A conflicting frozen window must fail as a whole.
    candidates = pd.read_parquet(train_path, columns=['pkg_no'], filters=[issue_filter], use_threads=False)
    candidate_ids = _package_keys(candidates['pkg_no'], 'AFT fit candidates')
    exact_ids = _package_keys(fit_ids, 'Frozen fit IDs')
    if exact_ids.empty or not exact_ids.isin(candidate_ids).all():
        raise ValueError('Frozen fit keys must be a nonempty subset of AFT fit candidates')
    for name, ids in [('calibration', cal_ids), ('evaluation', eval_ids), ('final holdout', holdout_ids)]:
        if candidate_ids.isin(_package_keys(ids, f'Frozen {name} IDs')).any():
            raise ValueError(f'AFT fit candidates overlap frozen {name} keys')

    # Only the yes/no cutoff state is needed for selection. Neither the future
    # signature timestamp nor the eventual target is projected for these rows.
    censored = pd.read_parquet(train_path, columns=list(SHARED_COLUMNS),
                               filters=[issue_filter, ('sign_time', '>=', label_cutoff)], use_threads=False)
    expected_ids = candidate_ids[~candidate_ids.isin(exact_ids)]
    actual_ids = _package_keys(censored['pkg_no'], 'AFT censored fit IDs')
    if not actual_ids.equals(expected_ids):
        raise ValueError('Cutoff signature status and frozen fit keys do not cover the ordered AFT fit candidates')
    censored['pkg_no'] = censored['pkg_no'].astype('string')
    return censored


def aft_fit_bounds(fit, censored, label_cutoff):
    """Build exact then right-censored bounds in model-matrix row order."""
    cutoff = _timestamp(label_cutoff, 'AFT fit label cutoff')
    sign = pd.to_datetime(fit['sign_time'], errors='coerce')
    exact = pd.to_numeric(fit['y_total_hours'], errors='coerce').to_numpy(dtype='float64')
    if (sign.empty or sign.dt.tz is not None or not sign.lt(cutoff).all()
            or not np.isfinite(exact).all() or (exact <= 0).any()):
        raise ValueError('AFT exact labels require positive finite hours signed strictly before the cutoff')
    if censored is None or censored.empty:
        return exact, exact.copy()

    pay = pd.to_datetime(censored['pay_time'], errors='coerce')
    issue = pd.to_datetime(censored['issue_time'], errors='coerce')
    if (pay.dt.tz is not None or issue.dt.tz is not None or not pay.le(issue).all()
            or not issue.lt(cutoff).all()):
        raise ValueError('AFT censored bounds require valid pay_time <= issue_time < cutoff')
    lower = (cutoff - pay).dt.total_seconds().to_numpy() / 3600
    return np.concatenate([exact, lower]), np.concatenate([exact, np.full(len(lower), np.inf)])

"""Official A/B score semantics plus strict key alignment and group diagnostics."""

import numpy as np
import pandas as pd


def read_prediction_csv(path):
    """Round-trip commitment hours across the evaluation environments."""
    return pd.read_csv(path, dtype={"pkg_no": "string"}, keep_default_na=False,
                       converters={"pred_hours": float})


def validate_predictions(items, predictions, exact_columns=True):
    required = ['pkg_no', 'pred_hours']
    if list(predictions.columns) != required and exact_columns:
        raise ValueError('Predictions must have exactly pkg_no,pred_hours in that order')
    if not set(required).issubset(predictions.columns) or 'pkg_no' not in items:
        raise ValueError('Required prediction or item columns are missing')
    if items.empty or predictions.empty:
        raise ValueError('Empty item or prediction set')
    for name, frame in [('items', items), ('predictions', predictions)]:
        ids = frame['pkg_no']
        if ids.isna().any() or ids.astype('string').str.strip().eq('').any():
            raise ValueError(f'{name} has missing/empty package keys')
        if ids.duplicated().any():
            raise ValueError(f'{name} has duplicate package keys')
        if not ids.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f'{name} keys must be strings; read pkg_no explicitly as string')
    if set(items['pkg_no']) != set(predictions['pkg_no']):
        raise ValueError('Prediction keys do not exactly match item keys')
    values = pd.to_numeric(predictions['pred_hours'], errors='coerce')
    if not (np.isfinite(values) & (values > 0)).all():
        raise ValueError('pred_hours must be finite and strictly positive')
    predictions = predictions[required].copy()
    predictions['pred_hours'] = values.astype('float64')
    aligned = items.merge(predictions, on='pkg_no', how='left', sort=False,
                          validate='one_to_one')
    if len(aligned) != len(items) or not np.array_equal(
            aligned['pkg_no'].to_numpy(dtype=object), items['pkg_no'].to_numpy(dtype=object)):
        raise ValueError('Prediction alignment changed row order or count')
    return aligned


def prediction_diagnostics(truth, predictions):
    """Return parcel-weighted overall diagnostics from the saved predictions."""
    aligned = validate_predictions(truth.reset_index(drop=True), predictions)
    actual = pd.to_numeric(aligned["y_total_hours"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(actual).all():
        raise ValueError("Truth contains nonfinite/non-numeric target values")
    promised = aligned["pred_hours"].to_numpy()
    residual = actual - promised
    return {"MAE": float(np.abs(residual).mean()), "mean_pred_hours": float(promised.mean()),
            "severe_rate": float((residual > 24).mean())}


def score_predictions(truth, predictions, allow_null_groups=False):
    if not {'lgs_company_nm', 'y_total_hours'}.issubset(truth.columns):
        raise ValueError('Truth requires lgs_company_nm and y_total_hours')
    aligned = validate_predictions(truth.reset_index(drop=True), predictions)
    y = pd.to_numeric(aligned['y_total_hours'], errors='coerce')
    if not np.isfinite(y).all():
        raise ValueError('Truth contains nonfinite/non-numeric target values')
    groups = aligned['lgs_company_nm']
    if groups.isna().any() and not allow_null_groups:
        raise ValueError('Null carrier requires official clarification; no label rewriting is allowed')
    d = pd.DataFrame({'group': groups, 'y': y, 'T': aligned['pred_hours']})
    d['ontime'] = d['y'] <= d['T']
    d['severe'] = (d['y'] - d['T']) > 24
    # dropna=True matches the official scorer; null groups require explicit opt-in.
    by_group = d.groupby('group', observed=True, dropna=True).agg(
        n=('y', 'size'), mean_pred=('T', 'mean'), mean_true=('y', 'mean'),
        ontime_rate=('ontime', 'mean'), severe_rate=('severe', 'mean'))
    by_group['pen_ontime'] = 500 * np.maximum(0, .90 - by_group['ontime_rate'])
    by_group['pen_severe'] = 500 * np.maximum(0, by_group['severe_rate'] - .03)
    me = float(abs(d['T'].mean() - d['y'].mean()))
    on = float(by_group['pen_ontime'].sum())
    severe = float(by_group['pen_severe'].sum())
    return {'ME': me, 'pen_ontime': on, 'pen_severe': severe,
            'B': me + on, 'A': me + on + severe}, by_group.reset_index()

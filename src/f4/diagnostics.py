"""Read-only diagnostics. These summaries never choose parameters or fit models."""

import numpy as np
import pandas as pd

from .metrics import score_predictions, validate_predictions
from .models import empirical_quantile, finite_vector


def cohort_diagnostics(cohorts, times, calibration_maturity=None):
    result = {'scope': 'Descriptive only; never used to change frozen cohorts or fit models',
              'maturity_denominator': 'Official eventually-signed data only; not the complete arrival population',
              'phases': {}}
    for phase, frame in cohorts.items():
        y = finite_vector(frame['y_total_hours'], 'target', positive=True)
        issue = pd.to_datetime(frame['issue_time'], errors='raise')
        summary = {'n': len(frame), 'mean_hours': float(y.mean()),
                   'q50_hours': empirical_quantile(y, .5),
                   'q90_hours': empirical_quantile(y, .9),
                   'q97_hours': empirical_quantile(y, .97),
                   'issue_min': issue.min(), 'issue_max': issue.max()}
        if phase in {'fit', 'cal'}:
            cutoff = pd.Timestamp(times['F' if phase == 'fit' else 'C'])
            ages = (cutoff - issue).dt.total_seconds().to_numpy() / 3600
            if (ages < 0).any():
                raise ValueError('Label cutoff precedes cohort issue times')
            summary.update(label_cutoff=cutoff, median_observation_hours=float(np.median(ages)),
                           selection='Only parcels signed before cutoff')
        else:
            summary['selection'] = 'All frozen evaluation parcels, including slow deliveries'
        summary['by_carrier'] = [
            {'group': str(group), 'n': len(rows), 'mean_hours': float(rows.y_total_hours.mean()),
             'q90_hours': empirical_quantile(rows.y_total_hours, .9),
             'q97_hours': empirical_quantile(rows.y_total_hours, .97)}
            for group, rows in frame.groupby('lgs_company_nm', observed=True)]
        result['phases'][phase] = summary
    if calibration_maturity is not None:
        result['calibration_maturity'] = calibration_maturity
    return result


def prediction_diagnostics(truth, predictions):
    aligned = validate_predictions(truth, predictions)
    metrics, groups = score_predictions(truth, predictions)
    y = finite_vector(aligned.y_total_hours, 'target', positive=True)
    pred = aligned.pred_hours.to_numpy()
    groups['total_penalty'] = groups.pen_ontime + groups.pen_severe
    return {'n': len(y), 'metrics': metrics,
            'mean_prediction_hours': float(pred.mean()), 'mean_truth_hours': float(y.mean()),
            'signed_mean_bias_hours': float((pred - y).mean()),
            'mae_hours_diagnostic_only': float(np.abs(pred - y).mean()),
            'overall_ontime_rate': float((y <= pred).mean()),
            'overall_severe_rate': float((y - pred > 24).mean()),
            'carriers_below_90pct_ontime': int((groups.ontime_rate < .9).sum()),
            'carriers_above_3pct_severe': int((groups.severe_rate > .03).sum()),
            'all_carriers_meet_targets': bool((groups.ontime_rate >= .9).all() and (groups.severe_rate <= .03).all()),
            'by_carrier': groups.sort_values('total_penalty', ascending=False).to_dict('records'),
            'interpretation': 'Overall coverage does not replace per-carrier thresholds; score change is not accuracy change'}

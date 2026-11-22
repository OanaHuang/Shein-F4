"""Post-hoc development diagnostics; never provides model inputs or offsets."""

import pandas as pd

from .features import validate_pkg_ids
from .models import empirical_quantile, finite_vector


def distribution(frame):
    if frame.empty:
        return {'n': 0, 'mean_hours': None, 'q90_hours': None, 'q97_hours': None}
    values = finite_vector(frame.y_total_hours, 'diagnostic target', positive=True)
    return {'n': len(frame), 'mean_hours': float(values.mean()),
            'q90_hours': empirical_quantile(values, .90),
            'q97_hours': empirical_quantile(values, .97)}


def audit_tail_queue(queue, evaluation, times):
    """Separate available/full-queue and full-queue/evaluation distribution gaps.

    The complete queue contains hindsight labels forbidden in as-of fitting.
    This function returns descriptive summaries only; no predictions/scores.
    """
    start, end, cutoff = (pd.Timestamp(times[k]) for k in ('F', 'E', 'C'))
    eval_end = cutoff + pd.Timedelta(days=times['H_days'])
    for frame in (queue, evaluation):
        validate_pkg_ids(frame.pkg_no, len(frame))
        finite_vector(frame.y_total_hours, 'diagnostic target', positive=True)
        if frame.lgs_company_nm.isna().any():
            raise ValueError('Missing carrier in tail audit')
        for column in ('issue_time', 'pay_time', 'sign_time'):
            parsed = pd.to_datetime(frame[column], errors='coerce')
            if parsed.isna().any():
                raise ValueError('Invalid diagnostic timestamp')
    issue = pd.to_datetime(queue.issue_time)
    eval_issue = pd.to_datetime(evaluation.issue_time)
    if not (issue.ge(start) & issue.lt(end)).all():
        raise ValueError('Queue outside fixed calibration issue interval')
    if not (eval_issue.ge(cutoff) & eval_issue.lt(eval_end)).all():
        raise ValueError('Evaluation outside frozen interval')
    if set(queue.pkg_no).intersection(evaluation.pkg_no):
        raise ValueError('Queue and evaluation keys overlap')
    available = pd.to_datetime(queue.sign_time).lt(cutoff)
    rows = []
    for group, group_queue in queue.groupby('lgs_company_nm', observed=True):
        group_available = available.loc[group_queue.index]
        mature = group_queue.loc[group_available]
        late = group_queue.loc[~group_available]
        evaluated = evaluation.loc[evaluation.lgs_company_nm.eq(group)]
        summaries = {'available_at_cutoff': distribution(mature),
                     'late_signed_hindsight_only': distribution(late),
                     'complete_queue_hindsight_only': distribution(group_queue),
                     'evaluation': distribution(evaluated)}
        gaps = {}
        if len(mature) and len(evaluated):
            for metric in ('mean_hours', 'q90_hours', 'q97_hours'):
                a = summaries['available_at_cutoff'][metric]
                q = summaries['complete_queue_hindsight_only'][metric]
                e = summaries['evaluation'][metric]
                gaps[metric] = {'queue_minus_available': q - a,
                                'evaluation_minus_queue': e - q,
                                'evaluation_minus_available': e - a}
        daily = []
        for day, block in group_queue.groupby(pd.to_datetime(group_queue.issue_time).dt.strftime('%Y-%m-%d')):
            visible = available.loc[block.index]
            daily.append({'issue_day': day, 'queue_n': len(block),
                          'available_n': int(visible.sum()),
                          'available_fraction': float(visible.mean()),
                          'available': distribution(block.loc[visible]),
                          'complete_queue_hindsight_only': distribution(block)})
        observation = (cutoff - pd.to_datetime(group_queue.pay_time)).dt.total_seconds() / 3600
        rows.append({'group': str(group), 'queue_n': len(group_queue),
                     'available_n': len(mature), 'late_signed_n': len(late),
                     'available_fraction': len(mature) / len(group_queue),
                     'pay_to_cutoff_hours': {'min': float(observation.min()),
                                             'median': float(observation.median()),
                                             'max': float(observation.max())},
                     'distributions': summaries, 'descriptive_gaps': gaps,
                     'by_issue_day': daily})
    return {'scope': 'Hindsight diagnostic only; no model fitting, tuning, predictions or new score',
            'denominator': 'Official eventually-signed records, not the complete arrival population',
            'gap_interpretation': 'Arithmetic descriptive decomposition, not causal attribution or an admissible calibration offset',
            'evaluation_only_carriers': sorted(set(evaluation.lgs_company_nm) - set(queue.lgs_company_nm)),
            'by_carrier': rows}

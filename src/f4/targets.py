"""Explicit total/remaining-hour targets; anomalous rows fail, never disappear."""

import numpy as np
import pandas as pd

from .models import finite_vector, positive_prediction


class TargetTransform:
    def __init__(self, kind='total'):
        if kind not in {'total', 'remaining'}:
            raise ValueError('target must be total or remaining')
        self.kind = kind

    def offset(self, frame):
        if self.kind == 'total':
            return np.zeros(len(frame))
        pay = pd.to_datetime(frame['pay_time'], errors='coerce')
        issue = pd.to_datetime(frame['issue_time'], errors='coerce')
        elapsed = (issue - pay).dt.total_seconds().to_numpy() / 3600
        if not np.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError('Remaining target requires valid nonnegative issue-minus-pay; no rows removed')
        return elapsed

    def transform(self, frame):
        target = finite_vector(frame['y_total_hours'], 'total hours', positive=True) - self.offset(frame)
        if (target <= 0).any():
            raise ValueError('Nonpositive remaining hours; audit timestamps before fitting; no rows removed')
        return pd.Series(target, index=frame.index)

    def restore(self, values, frame):
        prediction = finite_vector(values, 'target predictions')
        if len(prediction) != len(frame):
            raise ValueError('Target prediction row count differs')
        return positive_prediction(prediction + self.offset(frame))

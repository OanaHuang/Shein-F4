import unittest

import numpy as np
import pandas as pd

from f4.history_floor import apply_history_floor, fit_history_floor


class HistoryFloorTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame({'pkg_no': ['SMOKE_001', 'SMOKE_002', 'SMOKE_003'],
                             'lgs_company_nm': ['A', 'A', 'B'],
                             'issue_time': ['2026-08-01'] * 3,
                             'sign_time': ['2026-08-02'] * 3,
                             'y_total_hours': [10., 24., 20.]})

    def test_floor_does_not_shorten_and_sparse_unknown_keep_original(self):
        fit = fit_history_floor(self.history(), '2026-09-01', min_group_samples=2)
        actual = apply_history_floor([12., 30., 1., 2.], ['A', 'A', 'B', 'UNKNOWN'], fit)
        np.testing.assert_array_equal(actual, [24., 30., 1., 2.])

    def test_future_labels_recent_history_and_duplicate_keys_rejected(self):
        for column, value in [('sign_time', '2026-09-01'), ('issue_time', '2026-08-11'),
                              ('pkg_no', 'SMOKE_002')]:
            frame = self.history()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                fit_history_floor(frame, '2026-09-01')

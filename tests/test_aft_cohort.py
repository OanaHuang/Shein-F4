"""Synthetic AFT cohort and bound checks with no model fitting."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from f4.aft_cohort import aft_fit_bounds, read_censored_fit
from f4.features import SHARED_COLUMNS


ISSUE_CUTOFF = pd.Timestamp('2026-08-04')
LABEL_CUTOFF = pd.Timestamp('2026-08-08')


def fixture():
    rows = [
        ('0002', '2026-08-02', '2026-08-04', 48.),
        ('0007', '2026-08-03', '2026-08-11', 192.),
        ('0003', '2026-08-02', '2026-08-08', 144.),
        ('0001', '2026-08-01', '2026-08-04', 72.),
        ('0004', '2026-08-04', '2026-08-05', 24.),
        ('0005', '2026-08-08', '2026-08-20', 288.),
        ('0006', '2026-08-13', '2026-08-22', 216.),
    ]
    frame = pd.DataFrame(rows, columns=['pkg_no', 'pay_time', 'sign_time', 'y_total_hours'])
    frame['pay_time'] = pd.to_datetime(frame['pay_time'])
    frame['issue_time'] = frame['pay_time'].where(~frame['pkg_no'].isin(['0001', '0003']),
                                                 frame['pay_time'] + pd.Timedelta(hours=6))
    frame['sign_time'] = pd.to_datetime(frame['sign_time'])
    for name in set(SHARED_COLUMNS) - set(frame):
        frame[name] = 'synthetic'
    ids = {'fit_ids': ['0001', '0002'], 'cal_ids': ['0004'], 'eval_ids': ['0005'], 'holdout_ids': ['0006']}
    return frame, ids


def load_supplement(frame, ids):
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'train.parquet'
        frame.to_parquet(path, index=False)
        return read_censored_fit(path, **ids, issue_cutoff=ISSUE_CUTOFF, label_cutoff=LABEL_CUTOFF)


class AFTCohortTests(unittest.TestCase):
    def test_shared_supplement_and_pay_time_bounds_in_model_order(self):
        frame, ids = fixture()
        original_read = pd.read_parquet
        with patch('f4.aft_cohort.pd.read_parquet', wraps=original_read) as reader:
            censored = load_supplement(frame, ids)
        self.assertEqual(reader.call_args_list[0].kwargs['columns'], ['pkg_no'])
        self.assertEqual(reader.call_args_list[1].kwargs['columns'], list(SHARED_COLUMNS))
        self.assertEqual(censored['pkg_no'].tolist(), ['0007', '0003'])
        fit = frame.set_index('pkg_no').loc[ids['fit_ids']]
        lower, upper = aft_fit_bounds(fit, censored, LABEL_CUTOFF)
        np.testing.assert_array_equal(lower, [72., 48., 120., 144.])
        np.testing.assert_array_equal(upper, [72., 48., np.inf, np.inf])
        exact_lower, exact_upper = aft_fit_bounds(fit, None, LABEL_CUTOFF)
        np.testing.assert_array_equal(exact_lower, [72., 48.])
        np.testing.assert_array_equal(exact_upper, exact_lower)

    def test_future_and_holdout_label_values_do_not_change_bounds(self):
        frame, ids = fixture()
        original = load_supplement(frame, ids)
        changed = frame.copy()
        censored = changed['pkg_no'].isin(['0007', '0003'])
        changed.loc[censored, 'sign_time'] += pd.Timedelta(days=150)
        changed.loc[censored, 'y_total_hours'] = -9999.
        held_out = changed['pkg_no'].isin(ids['holdout_ids'])
        changed.loc[held_out, 'sign_time'] = pd.NaT
        changed.loc[held_out, 'y_total_hours'] = np.nan
        pd.testing.assert_frame_equal(original, load_supplement(changed, ids))

    def test_overlap_rejected_before_any_sign_time_filter(self):
        frame, ids = fixture()
        for phase in ('holdout_ids', 'cal_ids', 'eval_ids'):
            with self.subTest(phase=phase):
                original_read = pd.read_parquet
                with patch('f4.aft_cohort.pd.read_parquet', wraps=original_read) as reader:
                    with self.assertRaisesRegex(ValueError, 'overlap'):
                        load_supplement(frame, {**ids, phase: ['0003']})
                self.assertEqual(reader.call_count, 1)
                self.assertEqual(reader.call_args.kwargs['filters'], [('issue_time', '<', ISSUE_CUTOFF)])

    def test_inconsistent_candidate_labels_do_not_silently_shrink_cohort(self):
        frame, ids = fixture()
        for changed_sign in (pd.NaT, LABEL_CUTOFF - pd.Timedelta(hours=1)):
            with self.subTest(changed_sign=changed_sign):
                changed = frame.copy()
                changed.loc[changed['pkg_no'] == '0007', 'sign_time'] = changed_sign
                with self.assertRaisesRegex(ValueError, 'do not cover'):
                    load_supplement(changed, ids)

    def test_rejects_invalid_exact_and_censored_label_bounds(self):
        frame, ids = fixture()
        fit = frame.set_index('pkg_no').loc[ids['fit_ids']].copy()
        censored = load_supplement(frame, ids)
        bad_fit = fit.copy()
        bad_fit.loc['0001', 'sign_time'] = LABEL_CUTOFF
        with self.assertRaisesRegex(ValueError, 'signed strictly before'):
            aft_fit_bounds(bad_fit, censored, LABEL_CUTOFF)
        censored.loc[0, 'pay_time'] = LABEL_CUTOFF
        with self.assertRaisesRegex(ValueError, 'pay_time <= issue_time'):
            aft_fit_bounds(fit, censored, LABEL_CUTOFF)

"""Diagnose aggregate/individual tradeoffs without altering score semantics."""

import unittest
import pandas as pd

from f4.diagnostics import cohort_diagnostics, prediction_diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_overall_coverage_cannot_hide_small_carrier_failure(self):
        truth = pd.DataFrame({'pkg_no': [f'SMOKE_{i:03d}' for i in range(101)],
                              'lgs_company_nm': ['A'] * 100 + ['B'], 'y_total_hours': [40.] * 101})
        prediction = pd.DataFrame({'pkg_no': truth.pkg_no, 'pred_hours': [41.] * 100 + [1.]})
        result = prediction_diagnostics(truth, prediction)
        self.assertGreater(result['overall_ontime_rate'], .9)
        self.assertFalse(result['all_carriers_meet_targets'])
        self.assertEqual(result['carriers_below_90pct_ontime'], 1)
        self.assertEqual(result['carriers_above_3pct_severe'], 1)
        self.assertEqual(result['by_carrier'][0]['group'], 'B')
        self.assertAlmostEqual(result['metrics']['pen_ontime'], 450.)
        self.assertAlmostEqual(result['metrics']['pen_severe'], 485.)

    def test_mean_error_is_not_individual_error_and_keys_still_checked(self):
        truth = pd.DataFrame({'pkg_no': ['SMOKE_01', 'SMOKE_02'],
                              'lgs_company_nm': ['A', 'A'], 'y_total_hours': [10., 30.]})
        prediction = pd.DataFrame({'pkg_no': truth.pkg_no, 'pred_hours': [20., 20.]})
        result = prediction_diagnostics(truth, prediction)
        self.assertEqual(result['metrics']['ME'], 0.)
        self.assertEqual(result['mae_hours_diagnostic_only'], 10.)
        with self.assertRaises(ValueError):
            prediction_diagnostics(truth, prediction.iloc[:1])

    def test_cohort_summary_labels_selection_explicitly(self):
        frame = pd.DataFrame({'lgs_company_nm': ['A'], 'y_total_hours': [24.],
                              'issue_time': ['2026-08-01']})
        result = cohort_diagnostics({'fit': frame, 'cal': frame, 'eval': frame},
                                    {'F': '2026-08-02', 'C': '2026-08-03'})
        self.assertEqual(result['phases']['fit']['median_observation_hours'], 24.)
        self.assertEqual(result['phases']['cal']['median_observation_hours'], 48.)
        self.assertIn('including slow', result['phases']['eval']['selection'])

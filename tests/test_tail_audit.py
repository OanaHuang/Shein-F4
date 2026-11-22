import unittest

import pandas as pd

from f4.tail_audit import audit_tail_queue


class TailAuditTests(unittest.TestCase):
    def frames(self):
        queue = pd.DataFrame({'pkg_no': ['SMOKE_001', 'SMOKE_002'],
                              'lgs_company_nm': ['A', 'A'],
                              'issue_time': ['2026-08-01', '2026-08-02'],
                              'pay_time': ['2026-08-01', '2026-08-02'],
                              'sign_time': ['2026-08-02', '2026-08-05'],
                              'y_total_hours': [24., 72.]})
        evaluation = pd.DataFrame({'pkg_no': ['SMOKE_003'], 'lgs_company_nm': ['A'],
                                   'issue_time': ['2026-08-05'], 'pay_time': ['2026-08-05'],
                                   'sign_time': ['2026-08-09'], 'y_total_hours': [96.]})
        times = {'F': '2026-08-01', 'E': '2026-08-03', 'C': '2026-08-05', 'H_days': 2}
        return queue, evaluation, times

    def test_cutoff_is_strict_and_gaps_are_descriptive(self):
        row = audit_tail_queue(*self.frames())['by_carrier'][0]
        self.assertEqual(row['available_fraction'], .5)
        self.assertEqual(row['distributions']['available_at_cutoff']['q97_hours'], 24.)
        self.assertEqual(row['distributions']['complete_queue_hindsight_only']['q97_hours'], 72.)
        gap = row['descriptive_gaps']['q97_hours']
        self.assertEqual(gap['queue_minus_available'], 48.)
        self.assertEqual(gap['evaluation_minus_queue'], 24.)
        self.assertEqual(gap['evaluation_minus_available'], 72.)
        self.assertEqual(row['by_issue_day'][1]['available']['n'], 0)

    def test_missing_evaluation_group_is_not_a_fake_zero(self):
        queue, evaluation, times = self.frames()
        evaluation['lgs_company_nm'] = 'B'
        result = audit_tail_queue(queue, evaluation, times)
        self.assertEqual(result['evaluation_only_carriers'], ['B'])
        self.assertEqual(result['by_carrier'][0]['descriptive_gaps'], {})

    def test_keys_and_fixed_time_boundaries_fail_closed(self):
        queue, evaluation, times = self.frames()
        queue.loc[1, 'pkg_no'] = queue.loc[0, 'pkg_no']
        with self.assertRaises(ValueError):
            audit_tail_queue(queue, evaluation, times)
        queue, evaluation, times = self.frames()
        evaluation.loc[0, 'issue_time'] = '2026-08-07'
        with self.assertRaises(ValueError):
            audit_tail_queue(queue, evaluation, times)

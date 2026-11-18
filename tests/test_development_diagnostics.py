"""Small synthetic diagnostics checks; no real labels or model fitting."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('development_diagnostics_under_test',
                                            ROOT / 'scripts/diagnose_development.py')
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def synthetic_inputs(root):
    config = {'split_id': 'diagnostic_smoke',
              'development': [
                  {'name': 'dev_1', 'F': '2026-08-04', 'E': '2026-08-06', 'C': '2026-08-08', 'H_days': 2},
                  {'name': 'dev_2', 'F': '2026-08-06', 'E': '2026-08-08', 'C': '2026-08-10', 'H_days': 2}],
              'holdout': {'F': '2026-08-08', 'E': '2026-08-10', 'C': '2026-08-14', 'H_days': 2}}
    config_path = root / 'splits.yaml'
    config_path.write_text(yaml.safe_dump(config))
    split_id, holdout_spec, specs = diagnostics.validate_config(config)
    issue = pd.date_range('2026-08-01', periods=15, freq='D')
    sign = issue + pd.Timedelta(days=1)
    sign = pd.Series(sign)
    sign.iloc[1] = pd.Timestamp('2026-08-07')
    frame = pd.DataFrame({'pkg_no': [f'SMOKE_{i:03}' for i in range(15)],
                          'pay_time': issue - pd.Timedelta(hours=1), 'issue_time': issue,
                          'lgs_company_nm': ['A'] * 15,
                          'sign_time': sign.dt.strftime('%Y-%m-%d'),
                          'y_total_hours': ((sign - (issue - pd.Timedelta(hours=1))).dt.total_seconds() / 3600).astype(str)})
    isolated = frame['issue_time'].ge(holdout_spec.C) & frame['issue_time'].lt(holdout_spec.end)
    frame.loc[isolated, ['sign_time', 'y_total_hours']] = 'HOLDOUT_MUST_NOT_BE_PARSED'
    data_dir = root / 'data'
    data_dir.mkdir()
    train = data_dir / 'train.parquet'
    frame.to_parquet(train, index=False)
    split_root = root / 'artifacts'
    split_root.mkdir()

    def save_keys(path, keys):
        path.parent.mkdir(parents=True, exist_ok=True)
        keys.to_parquet(path, index=False)
        return {'n': len(keys), 'file_sha256': diagnostics.sha256_file(path),
                'keys_sha256': diagnostics._key_digest(keys)}

    canonical = {'split_id': split_id, 'holdout': holdout_spec.metadata(),
                 'development': [spec.metadata() for spec in specs]}
    report = {'status': 'complete', 'holdout_isolation_checked': True, 'split_id': split_id,
              'config': canonical,
              'config_sha256': hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
              'config_file_sha256': diagnostics.sha256_file(config_path),
              'train_parquet_sha256': diagnostics.sha256_file(train),
              'holdout': save_keys(split_root / 'holdout_ids.parquet', frame.loc[isolated, ['pkg_no']]),
              'folds': {}}
    known = frame.loc[~isolated].copy()
    known['sign_time'] = pd.to_datetime(known['sign_time'])
    for spec in specs:
        masks = {'fit': known['issue_time'].lt(spec.F) & known['sign_time'].lt(spec.F),
                 'cal': known['issue_time'].ge(spec.F) & known['issue_time'].lt(spec.E) & known['sign_time'].lt(spec.C),
                 'eval': known['issue_time'].ge(spec.C) & known['issue_time'].lt(spec.end)}
        report['folds'][spec.name] = {'times': spec.metadata(), 'artifacts': {
            phase: save_keys(split_root / spec.name / f'{phase}_ids.parquet', known.loc[mask, ['pkg_no']])
            for phase, mask in masks.items()}}
    report_path = root / 'split_report.json'
    report_path.write_text(json.dumps(report))
    return data_dir, config_path, report_path, split_root, set(frame.loc[isolated, 'pkg_no'])


class DevelopmentDiagnosticTests(unittest.TestCase):
    def test_labels_are_filtered_before_holdout_can_be_materialized(self):
        with TemporaryDirectory() as directory:
            data, config, report, keys, isolated = synthetic_inputs(Path(directory))
            calls = []

            def reader(path, **kwargs):
                calls.append(kwargs)
                if 'y_total_hours' in kwargs['columns'] or 'sign_time' in kwargs['columns']:
                    selected = kwargs['filters'][0][2]
                    self.assertFalse(set(selected).intersection(isolated))
                    self.assertEqual(kwargs['filters'][0][:2], ('pkg_no', 'in'))
                return pd.read_parquet(path, **kwargs)

            specs, cohorts, provenance = diagnostics.load_development(data, config, report, keys, reader=reader)
            self.assertEqual(calls[0]['columns'], diagnostics.FEATURE_COLUMNS)
            self.assertEqual(calls[1]['columns'], diagnostics.LABEL_COLUMNS)
            self.assertEqual(len(calls), 2)
            self.assertEqual(provenance['excluded_holdout_rows'], 2)
            self.assertEqual(provenance['candidate_label_rows'], provenance['filtered_label_rows'])
            self.assertEqual(len(cohorts['dev_1']['fit']), 3)
            self.assertTrue(all(not set(frame['pkg_no']).intersection(isolated)
                                for phases in cohorts.values() for frame in phases.values()))

    def test_maturity_denominator_includes_unavailable_labels(self):
        frame = pd.DataFrame({'issue_time': pd.to_datetime(['2026-08-01'] * 3),
                              'lgs_company_nm': ['A'] * 3,
                              'sign_time': pd.to_datetime(['2026-08-02', '2026-08-03', '2026-08-08']),
                              'y_total_hours': [24., 48., 168.]})
        row = diagnostics.maturity_rows(frame, 'dev_1', 'fit', pd.Timestamp('2026-08-04'))[0]
        self.assertEqual(row['queue_n'], 3)
        self.assertEqual(row['available_n'], 2)
        self.assertAlmostEqual(row['available_rate'], 2 / 3)
        self.assertEqual(row['available_mean_hours'], 36.)
        self.assertEqual(row['retrospective_full_mean_hours'], 80.)

    def test_missing_prediction_fails_instead_of_dropping_a_package(self):
        frame = pd.DataFrame({'pkg_no': ['SMOKE_1', 'SMOKE_2'],
                              'pay_time': pd.to_datetime(['2026-08-01', '2026-08-02']),
                              'issue_time': pd.to_datetime(['2026-08-01 01:00', '2026-08-02 07:00']),
                              'y_total_hours': [24., 48.]})
        prediction = pd.DataFrame({'pkg_no': ['SMOKE_1'], 'pred_hours': [25.]})
        with self.assertRaisesRegex(ValueError, 'exactly match'):
            diagnostics.feature_bias_rows(frame, prediction, 'dev_1', 'M1')
        prediction = pd.DataFrame({'pkg_no': ['SMOKE_2', 'SMOKE_1'], 'pred_hours': [50., 25.]})
        rows = diagnostics.feature_bias_rows(frame, prediction, 'dev_1', 'M1')
        for feature in ['pay_weekday', 'issue_weekday', 'elapsed_hours']:
            self.assertEqual(sum(row['n'] for row in rows if row['feature'] == feature), 2)

    def test_source_change_fails_before_reading_labels(self):
        with TemporaryDirectory() as directory:
            data, config, report, keys, _ = synthetic_inputs(Path(directory))
            config.write_text(config.read_text() + '\n')
            reads = []
            with self.assertRaisesRegex(ValueError, 'provenance'):
                diagnostics.load_development(data, config, report, keys,
                                             reader=lambda *a, **kw: reads.append(kw))
            self.assertFalse(reads)


if __name__ == '__main__':
    unittest.main()

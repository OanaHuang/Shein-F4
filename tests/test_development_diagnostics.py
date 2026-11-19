"""Small synthetic diagnostics checks; no real labels or model fitting."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('development_diagnostics_under_test',
                                            ROOT / 'scripts/diagnose_development.py')
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)
from f4.splits import make_splits


def synthetic_inputs(root, *, protocol='asof_backtest'):
    root.mkdir(parents=True, exist_ok=True)
    config = {'split_id': 'diagnostic_smoke',
              'development': [
                  {'name': 'dev_1', 'F': '2026-08-04', 'E': '2026-08-06', 'C': '2026-08-08', 'H_days': 2},
                  {'name': 'dev_2', 'F': '2026-08-06', 'E': '2026-08-08', 'C': '2026-08-10', 'H_days': 2}],
              'holdout': {'F': '2026-08-08', 'E': '2026-08-10', 'C': '2026-08-14', 'H_days': 2}}
    config_path = root / 'legacy_splits.yaml'
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
    artifact_base = root / 'artifacts'
    report_path = root / 'legacy_report.json'
    make_splits(data_dir, config_path, artifact_base, report_path)
    split_root = artifact_base / split_id
    if protocol == 'asof_deploy_C_v1':
        current = deepcopy(config)
        current.update(split_id='diagnostic_deploy_c', protocol=protocol,
                       reference={'split_id': split_id, 'report': str(report_path),
                                  'report_sha256': diagnostics.sha256_file(report_path),
                                  'artifacts': str(split_root)})
        for value in [current['holdout'], *current['development']]:
            for field in ['model_fit_time', 'fit_label_cutoff', 'cal_label_cutoff']:
                value[field] = value['C']
        config_path = root / 'deploy_c_splits.yaml'
        config_path.write_text(yaml.safe_dump(current))
        report_path = root / 'deploy_c_report.json'
        make_splits(data_dir, config_path, artifact_base, report_path)
        split_root = artifact_base / current['split_id']
    return data_dir, config_path, report_path, split_root, set(frame.loc[isolated, 'pkg_no'])


def completed_prediction_fixture(root, cohorts, provenance):
    report_base = root / 'runs'
    jobs = []
    for fold, phases in cohorts.items():
        for model in ['M1', 'M2']:
            run_id = f'SMOKE_{fold}_{model}'
            run_dir = report_base / run_id
            run_dir.mkdir(parents=True)
            predictions = phases['eval'][['pkg_no']].copy()
            predictions['pred_hours'] = phases['eval']['y_total_hours'] + 1
            prediction = run_dir / 'predictions.csv'
            predictions.to_csv(prediction, index=False)
            manifest = {'status': 'complete', 'synthetic': False, 'fold': fold,
                        'experiment': model, 'source': provenance,
                        'predictions': {'path': str(prediction),
                                        'sha256': diagnostics.sha256_file(prediction)}}
            (run_dir / 'run_manifest.json').write_text(json.dumps(manifest))
            jobs.append({'run_id': run_id})
    path = root / 'batch_status.json'
    path.write_text(json.dumps({'status': 'complete', 'jobs': jobs}))
    return path, report_base


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

    def test_deploy_c_diagnostic_uses_C_cutoff_in_saved_maturity_output(self):
        results = {}
        with TemporaryDirectory() as directory:
            for protocol in ['asof_backtest', 'asof_deploy_C_v1']:
                root = Path(directory) / protocol
                data, config, report, keys, isolated = synthetic_inputs(root, protocol=protocol)
                specs, cohorts, provenance = diagnostics.load_development(data, config, report, keys)
                status, report_base = completed_prediction_fixture(root, cohorts, provenance)
                out = root / 'diagnostic_output'
                summary = diagnostics.diagnose(data, config, report, keys, status, out,
                                               report_base=report_base)
                maturity = pd.read_csv(out / 'maturity_by_date_carrier.csv')
                fit = maturity[maturity['fold'].eq('dev_1') & maturity['phase'].eq('fit')]
                results[protocol] = {'queue_n': int(fit['queue_n'].sum()),
                                     'available_n': int(fit['available_n'].sum()),
                                     'cutoffs': fit['cutoff_time'].unique().tolist()}
                self.assertTrue(summary['retrospective_diagnostic_only'])
                self.assertFalse(summary['holdout_labels_read'])
                self.assertFalse(summary['training_performed'])
                self.assertTrue(all(not set(frame['pkg_no']).intersection(isolated)
                                    for phases in cohorts.values() for frame in phases.values()))
            self.assertEqual(results['asof_backtest'], {
                'queue_n': 3, 'available_n': 1, 'cutoffs': ['2026-08-04T00:00:00']})
            self.assertEqual(results['asof_deploy_C_v1'], {
                'queue_n': 3, 'available_n': 3, 'cutoffs': ['2026-08-08T00:00:00']})

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

    def test_changed_pinned_reference_report_fails_before_diagnostic_label_read(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, config, report, keys, _ = synthetic_inputs(root, protocol='asof_deploy_C_v1')
            _, cohorts, provenance = diagnostics.load_development(data, config, report, keys)
            status, report_base = completed_prediction_fixture(root, cohorts, provenance)
            reference = yaml.safe_load(config.read_text())['reference']
            pinned_report = Path(reference['report'])
            before = pinned_report.read_bytes()
            # JSON meaning stays unchanged, but the pinned byte hash must fail.
            pinned_report.write_bytes(before + b'\n')
            self.assertEqual(json.loads(before), json.loads(pinned_report.read_bytes()))
            self.assertNotEqual(diagnostics.sha256_file(pinned_report), reference['report_sha256'])
            original_read = pd.read_parquet
            reads = []

            def tracked_read(*args, **kwargs):
                reads.append(kwargs.get('columns', []))
                return original_read(*args, **kwargs)

            out = root / 'rejected_diagnostic'
            with patch.object(diagnostics.pd, 'read_parquet', side_effect=tracked_read):
                with self.assertRaises(ValueError):
                    diagnostics.diagnose(data, config, report, keys, status, out,
                                         report_base=report_base)
            self.assertFalse(any({'sign_time', 'y_total_hours'}.intersection(columns)
                                 for columns in reads),
                             'Changed pinned reference was detected after reading labels')
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()

"""Survival model report checks with saved synthetic outputs and no model fitting."""

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('aft_comparison_under_test',
                                              ROOT / 'scripts/compare_deploy_C_aft.py')
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)

from f4.splits import _write_keys_once


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def comparison_fixture(root):
    plan = {'baseline_runs': {}, 'jobs': []}
    sample = pd.DataFrame({'pkg_no': ['0001', '0002', '0011', '0012', '0099'],
                           'lgs_company_nm': ['A'] * 5,
                           'y_total_hours': [25., 49., 25., 49., 9999.]})
    train_path = root / 'data/processed/core/train.parquet'
    train_path.parent.mkdir(parents=True)
    sample.to_parquet(train_path, index=False)
    split_id = 'provided_deploy_C_split'
    split_root = root / 'artifacts/splits' / split_id
    holdout = _write_keys_once(split_root / 'holdout_ids.parquet', sample.iloc[[4]][['pkg_no']])
    report = {'split_id': split_id, 'status': 'complete', 'holdout_isolation_checked': True,
              'train_parquet_sha256': comparison.sha256_file(train_path),
              'config_sha256': 'one-pinned-config', 'holdout': holdout, 'folds': {}}
    scorer = root / 'official/tools/score.py'
    scorer.parent.mkdir(parents=True)
    scorer.write_text('local scorer fixture', encoding='utf-8')
    manifests = {}
    for pos, fold in enumerate(comparison.DEVELOPMENT_FOLDS):
        fold_keys = split_root / fold
        truth = sample.iloc[pos * 2:pos * 2 + 2].reset_index(drop=True)
        cohorts = {'eval': _write_keys_once(fold_keys / 'eval_ids.parquet', truth[['pkg_no']])}
        for phase in ('fit', 'cal'):
            keys = pd.DataFrame({'pkg_no': [f'{fold}_{phase}']})
            cohorts[phase] = _write_keys_once(fold_keys / f'{phase}_ids.parquet', keys)
        times = {'C': f'2026-09-0{pos + 1}T00:00:00'}
        report['folds'][fold] = {'times': times, 'artifacts': cohorts}
        ids = [f'{fold}_{kind}' for kind in ('reference', 'exact', 'right_censored')]
        plan['baseline_runs'][fold] = ids[0]
        for index, run_id in enumerate(ids):
            experiment = 'M1' if index == 0 else 'AFT'
            config = {'experiment': experiment, 'features': 'shared_fields_v1', 'model': {'seed': 7}}
            predictions = truth[['pkg_no']].assign(pred_hours=25. + index)
            pred_path = root / 'outputs' / run_id / 'predictions.csv'
            pred_path.parent.mkdir(parents=True)
            predictions.to_csv(pred_path, index=False)
            metrics, _ = comparison.score_predictions(truth, predictions)
            manifest = {'status': 'complete', 'synthetic': False, 'run_id': run_id, 'fold': fold,
                        'protocol': comparison.DEPLOY_C_PROTOCOL, 'experiment': experiment,
                        'official_scorer_sha256': comparison.sha256_file(scorer),
                        'metrics': metrics, 'official_metrics': metrics, 'config': config,
                        'predictions': {'path': str(pred_path),
                                        'sha256': comparison.sha256_file(pred_path)},
                        'source': {'times': times, 'cohort_keys': dict(cohorts)}}
            if index:
                config['aft'] = {'include_right_censored': index == 2}
                config_path = root / f'{run_id}.yaml'
                config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
                plan['jobs'].append({'fold': fold, 'run_id': run_id, 'config': str(config_path)})
                manifest['aft_training'] = {'exact_n': 1, 'right_censored_n': index - 1,
                                            'training_n': index, 'encoder_fit_ids_sha256': fold}
                manifest['evaluation_diagnostics'] = comparison.prediction_diagnostics(truth, predictions)
                if index == 2:
                    keys = pd.DataFrame({'pkg_no': [f'{fold}_added']})
                    manifest['source']['cohort_keys']['aft_censored_fit'] = _write_keys_once(
                        root / 'artifacts' / run_id / 'censored_fit.parquet', keys)
            manifests[run_id] = manifest
    report_path = root / 'reports/splits' / f'{split_id}.json'
    write_json(report_path, report)
    for run_id, manifest in manifests.items():
        manifest['source'].update(split_id=split_id,
                                  split_report_sha256=comparison.sha256_file(report_path),
                                  train_parquet_sha256=report['train_parquet_sha256'],
                                  split_config_sha256=report['config_sha256'])
        write_json(root / 'reports/runs' / run_id / 'run_manifest.json', manifest)
    plan_path = root / 'plan.yaml'
    plan_path.write_text(yaml.safe_dump(plan), encoding='utf-8')
    return plan_path


def add_ngboost_references(root, aft_plan_path):
    plan = {'aft_reference_plan': str(aft_plan_path.relative_to(root)), 'jobs': []}
    aft_plan = yaml.safe_load(aft_plan_path.read_text(encoding='utf-8'))
    for job in aft_plan['jobs']:
        source_run_id = job['run_id']
        run_id = source_run_id + '_ngboost'
        source_manifest = root / 'reports/runs' / source_run_id / 'run_manifest.json'
        manifest = comparison.read_json(source_manifest)
        manifest.update(run_id=run_id, experiment='NGBOOST')
        config = manifest['config']
        config.update(experiment='NGBOOST', model={'minibatch_frac': .25}, ngboost=config.pop('aft'))
        manifest['ngboost_training'] = manifest.pop('aft_training')
        cohort_keys = manifest['source']['cohort_keys']
        if 'aft_censored_fit' in cohort_keys:
            record = cohort_keys.pop('aft_censored_fit')
            keys = pd.read_parquet(record['path'], columns=['pkg_no'], use_threads=False)
            cohort_keys['ngboost_censored_fit'] = _write_keys_once(
                root / 'artifacts' / run_id / 'censored_fit.parquet', keys)
        config_path = root / f'{run_id}.yaml'
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
        plan['jobs'].append({'fold': job['fold'], 'run_id': run_id, 'config': str(config_path)})
        write_json(root / 'reports/runs' / run_id / 'run_manifest.json', manifest)
    plan_path = root / 'ngboost_plan.yaml'
    plan_path.write_text(yaml.safe_dump(plan), encoding='utf-8')
    return plan_path


class AFTComparisonTests(unittest.TestCase):
    def test_comparison_only_reads_labels_for_frozen_baseline_evaluation_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan = comparison_fixture(root)
            parquet_reader = pd.read_parquet
            label_reads = []

            def read(path, **kwargs):
                if 'y_total_hours' in kwargs['columns']:
                    label_reads.append(kwargs['filters'])
                return parquet_reader(path, **kwargs)

            with patch.object(comparison.pd, 'read_parquet', side_effect=read):
                report = comparison.compare(plan, root)
            self.assertEqual(label_reads, [[('pkg_no', 'in', ['0001', '0002'])],
                                           [('pkg_no', 'in', ['0011', '0012'])]])
            rows = report['results']
            self.assertEqual([row['model'] for row in rows], list(comparison.MODEL_NAMES) * 2)
            self.assertEqual(rows[0]['MAE'], 12.0)
            self.assertEqual(rows[0]['mean_pred_hours'], 25.0)
            self.assertEqual(rows[0]['severe_rate'], 0.0)
            self.assertEqual(rows[2]['right_censored_fit_n'], 1)

    def test_ngboost_plan_compares_both_families_and_rejects_mismatched_keys(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            aft_plan = comparison_fixture(root)
            plan = add_ngboost_references(root, aft_plan)
            rows = comparison.compare(plan, root)['results']
            names = comparison.MODEL_NAMES + comparison.SURVIVAL_NAMES['NGBOOST']
            self.assertEqual([row['model'] for row in rows], list(names) * 2)
            self.assertEqual([row['right_censored_fit_n'] for row in rows], [0, 0, 1, 0, 1] * 2)
            self.assertEqual([row['fold'] for row in rows], ['dev_1'] * 5 + ['dev_2'] * 5)

            manifest_path = root / 'reports/runs/dev_1_right_censored_ngboost/run_manifest.json'
            manifest = comparison.read_json(manifest_path)
            keys = pd.DataFrame({'pkg_no': ['different-pkg']})
            manifest['source']['cohort_keys']['ngboost_censored_fit'] = _write_keys_once(
                root / 'artifacts/dev_1_right_censored_ngboost/changed.parquet', keys)
            write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, 'different right-censored fit keys'):
                comparison.compare(plan, root)


if __name__ == '__main__':
    unittest.main()

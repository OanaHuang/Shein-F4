"""Only synthetic end-to-end backtests and preparation authorization checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import subprocess
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import f4.common as common
from f4.backtest import run_backtest
from f4.features import SHARED_COLUMNS
from f4.models import load_model
from f4.safety import TrainingNotAuthorizedError
from f4.splits import make_splits


def synthetic_project(root, *, calibrated_baseline_fixture=False, future_unavailable_target=None,
                      deployment_extra_fit=False):
    rows = [
        ("2026-08-01", "2026-08-02"), ("2026-08-02", "2026-08-03"),
        ("2026-08-03", "2026-08-06"), ("2026-08-04", "2026-08-08"),
        ("2026-08-05", "2026-08-12"), ("2026-08-08", "2026-08-30"),
        ("2026-08-09", "2026-08-11"), ("2026-08-12", "2026-08-15"),
        ("2026-08-13", "2026-08-16"), ("2026-08-14", "2026-08-17"),
    ]
    if calibrated_baseline_fixture:
        # Both carriers have fitting and independent available calibration rows.
        # Their fitting constants differ, so translation equivalence is nontrivial.
        rows[1] = ("2026-08-02", "2026-08-02")
        rows[3] = ("2026-08-04", "2026-08-05")
    if future_unavailable_target is not None:
        rows[1] = ("2026-08-02", "2026-08-20")
    source = pd.DataFrame("KNOWN", index=range(len(rows)), columns=SHARED_COLUMNS)
    source["pkg_no"] = [f"SMOKE_{index:04d}" for index in range(len(rows))]
    source["pkg_goods_cnt"], source["weight"] = 1.0, 0.5
    source["issue_time"] = pd.to_datetime([row[0] for row in rows])
    source["sign_time"] = pd.to_datetime([row[1] for row in rows])
    source["pay_time"] = source["issue_time"] - pd.Timedelta(hours=1)
    source["y_total_hours"] = (source["sign_time"] - source["pay_time"]).dt.total_seconds() / 3600
    if future_unavailable_target is not None:
        source.loc[1, 'y_total_hours'] = future_unavailable_target
    source["lgs_company_nm"] = ["A", "B"] * 5
    if deployment_extra_fit:
        extra = source.iloc[[0]].copy()
        extra['pkg_no'] = 'SMOKE_C_EXTRA_FIT'
        extra['issue_time'] = pd.Timestamp('2026-08-02')
        extra['pay_time'] = pd.Timestamp('2026-08-01 23:00:00')
        extra['sign_time'] = pd.Timestamp('2026-08-07')
        extra['y_total_hours'] = 121.0
        source = pd.concat([source, extra], ignore_index=True)
    data_dir = root / "data"
    data_dir.mkdir()
    source.to_parquet(data_dir / "train.parquet", index=False)
    splits = {"split_id": "smoke_split", "holdout": {"F": "2026-08-04", "E": "2026-08-06", "C": "2026-08-12", "H_days": 2},
              "development": [{"name": "dev_1", "F": "2026-08-03", "E": "2026-08-05", "C": "2026-08-08", "H_days": 2}]}
    split_path = root / "splits.yaml"
    split_path.write_text(yaml.safe_dump(splits))
    make_splits(data_dir, split_path, root / "artifacts/splits", root / "reports/splits/smoke_split.json")
    return {"experiment": "B1", "protocol": "asof_backtest", "data_dir": str(data_dir),
            "splits": str(split_path), "artifacts_base": str(root / "artifacts"),
            "report_base": str(root / "reports"), "output_base": str(root / "outputs"),
            "model": {"kind": "carrier_constant", "min_group_samples": 1},
            "execution": {"stage": "preparation", "allow_real_training": False}}


def deployment_c_config(root, original):
    from f4.common import sha256_file
    old = yaml.safe_load(Path(original['splits']).read_text())
    updated = deepcopy(old)
    updated.update(split_id='smoke_deploy_C', protocol='asof_deploy_C_v1',
                   reference={'split_id': old['split_id'],
                              'report': str(root / 'reports/splits' / f"{old['split_id']}.json"),
                              'report_sha256': sha256_file(root / 'reports/splits' / f"{old['split_id']}.json"),
                              'artifacts': str(root / 'artifacts/splits' / old['split_id'])})
    for fold in [*updated['development'], updated['holdout']]:
        fold.update(model_fit_time=fold['C'], fit_label_cutoff=fold['C'], cal_label_cutoff=fold['C'])
    path = root / 'splits_deploy_C.yaml'
    path.write_text(yaml.safe_dump(updated))
    make_splits(Path(original['data_dir']), path, root / 'artifacts/splits',
                root / 'reports/splits/smoke_deploy_C.json')
    candidate = deepcopy(original)
    candidate.update(protocol='asof_deploy_C_v1', splits=str(path))
    return candidate


class BacktestSmokeTests(unittest.TestCase):
    def test_aft_dispatch_keeps_frozen_keys_and_passes_bounds_without_fitting(self):
        self.check_survival_dispatch("aft", "XGBoostAFTModel")

    def test_ngboost_dispatch_keeps_frozen_keys_and_passes_bounds_without_fitting(self):
        self.check_survival_dispatch("ngboost", "NGBoostSurvivalModel")

    def check_survival_dispatch(self, section, model_class):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_config = synthetic_project(root, future_unavailable_target=1_000_000,
                                           deployment_extra_fit=True)
            config = deployment_c_config(root, old_config)
            config.update(experiment=section.upper(), model={"n_estimators": 3, "n_jobs": 1})
            runs, calls = [], []

            def inspect_fit(model, matrix, lower, upper, *, pkg_ids, encoder_fit_ids_sha256):
                ids = pkg_ids.tolist()
                self.assertEqual(matrix.attrs["feature_fit_ids_sha256"], encoder_fit_ids_sha256)
                calls.append((ids, lower.tolist(), upper.tolist()))
                model.training_ids_sha256_ = matrix.attrs["pkg_ids_sha256"]
                return model

            model_path = f"f4.{section}_model.{model_class}"
            with patch(f"{model_path}.fit", autospec=True, side_effect=inspect_fit), \
                    patch(f"{model_path}.predict", autospec=True,
                          side_effect=lambda model, matrix: [24.] * len(matrix)):
                for suffix, include in [("EXACT", False), ("CENSORED", True)]:
                    config[section] = {"include_right_censored": include}
                    run_id = f"SMOKE_{section.upper()}_{suffix}"
                    runs.append(run_backtest(deepcopy(config), "dev_1", run_id, synthetic=True))
            exact, censored = runs
            self.assertEqual(calls[0], (["SMOKE_0000", "SMOKE_C_EXTRA_FIT"], [25., 121.], [25., 121.]))
            self.assertEqual(calls[1], (calls[0][0] + ["SMOKE_0001"], [25., 121., 145.],
                                        [25., 121., float("inf")]))
            for phase in ("fit", "cal", "eval"):
                self.assertEqual(exact["source"]["cohort_keys"][phase], censored["source"]["cohort_keys"][phase])
            training_key = f"{section}_training"
            self.assertEqual(exact[training_key]["encoder_fit_ids_sha256"],
                             censored[training_key]["encoder_fit_ids_sha256"])
            self.assertEqual(exact["artifacts"]["encoder"]["sha256"], censored["artifacts"]["encoder"]["sha256"])
            self.assertEqual(censored[training_key]["training_n"], 3)
            self.assertEqual(censored["evaluation_diagnostics"],
                             {"MAE": 265., "mean_pred_hours": 24., "severe_rate": 1.})
            artifact = censored["source"]["cohort_keys"][f"{section}_censored_fit"]
            self.assertEqual(pd.read_parquet(artifact["path"])["pkg_no"].tolist(), ["SMOKE_0001"])

    def test_survival_preparation_configs_reject_real_training_before_inputs(self):
        for section in ("aft", "ngboost"):
            for suffix in ("exact", "censored"):
                path = ROOT / "configs" / "training" / f"deploy_C_{section}_{suffix}.yaml"
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                for fold in ("dev_1", "dev_2"):
                    with self.subTest(config=path.name, fold=fold), patch("f4.backtest._prepare_inputs") as prepare:
                        with self.assertRaises(TrainingNotAuthorizedError):
                            run_backtest(config, fold, "unused", execute_training=True)
                        prepare.assert_not_called()

    def test_deployment_c_admits_available_old_fit_labels_and_keeps_cal_eval(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old_config = synthetic_project(root, calibrated_baseline_fixture=True,
                                           deployment_extra_fit=True)
            new_config = deployment_c_config(root, old_config)
            old_config.update(experiment='B1C', calibration={'min_group_samples': 1})
            new_config.update(experiment='B1C', calibration={'min_group_samples': 1})
            old = run_backtest(old_config, 'dev_1', 'SMOKE_F_CUTOFF', synthetic=True)
            new = run_backtest(new_config, 'dev_1', 'SMOKE_C_CUTOFF', synthetic=True)
            self.assertEqual(old['cohort_n']['fit'], 2)
            self.assertEqual(new['cohort_n']['fit'], 3)
            self.assertEqual(load_model(old['artifacts']['model']['path']).predict(['A', 'B']).tolist(), [25.0, 1.0])
            self.assertEqual(load_model(new['artifacts']['model']['path']).predict(['A', 'B']).tolist(), [121.0, 1.0])
            self.assertEqual(old['cohort_n']['cal'], new['cohort_n']['cal'])
            self.assertEqual(old['cohort_n']['eval'], new['cohort_n']['eval'])
            self.assertEqual(new['protocol'], 'asof_deploy_C_v1')
            self.assertEqual(new['source']['times']['model_fit_time'], '2026-08-08T00:00:00')
            for phase in ['cal', 'eval']:
                self.assertEqual(old['source']['cohort_keys'][phase]['file_sha256'],
                                 new['source']['cohort_keys'][phase]['file_sha256'])
            old_pred = pd.read_csv(old['predictions']['path'])
            new_pred = pd.read_csv(new['predictions']['path'])
            pd.testing.assert_frame_equal(old_pred, new_pred, check_exact=False, atol=1e-10, rtol=0)

    def test_deployment_c_unavailable_target_poison_does_not_change_predictions(self):
        predictions = []
        for poison in [-9999.0, 9999999.0]:
            with TemporaryDirectory() as directory:
                root = Path(directory)
                config = deployment_c_config(root, synthetic_project(root, future_unavailable_target=poison))
                config.update(experiment='B1C', calibration={'min_group_samples': 1})
                result = run_backtest(config, 'dev_1', 'SMOKE_C_POISON', synthetic=True)
                self.assertEqual(result['cohort_n']['fit'], 1)
                predictions.append(pd.read_csv(result['predictions']['path']))
        pd.testing.assert_frame_equal(predictions[0], predictions[1])

    def test_protocol_mismatch_is_rejected_before_label_read(self):
        import f4.backtest as implementation
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = deployment_c_config(root, synthetic_project(root))
            config['protocol'] = 'asof_backtest'
            reads = []
            original = pd.read_parquet

            def reader(*args, **kwargs):
                reads.append(kwargs.get('columns', []))
                return original(*args, **kwargs)

            with patch.object(implementation.pd, 'read_parquet', side_effect=reader):
                with self.assertRaisesRegex(ValueError, 'protocol does not match'):
                    run_backtest(config, 'dev_1', 'SMOKE_PROTOCOL_MISMATCH', synthetic=True)
            self.assertFalse(any('sign_time' in columns or 'y_total_hours' in columns for columns in reads))
        with patch.object(implementation.pd, 'read_parquet') as read:
            with self.assertRaisesRegex(ValueError, 'Unknown backtest protocol'):
                run_backtest({'protocol': 'unknown', 'experiment': 'B1'}, 'dev_1',
                             'SMOKE_UNKNOWN_PROTOCOL', synthetic=True)
            read.assert_not_called()

    def test_deployment_reference_changes_rejected_by_backtest_and_cli_before_labels(self):
        import f4.backtest as implementation
        from f4.common import sha256_file
        specification = importlib.util.spec_from_file_location('f4_backtest_reference_test', ROOT / 'scripts/backtest.py')
        cli = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cli)
        for mutation in ['report', 'old_fit', 'inherited_cal']:
            with self.subTest(mutation=mutation), TemporaryDirectory() as directory:
                root = Path(directory)
                config = deployment_c_config(root, synthetic_project(root))
                pd.DataFrame({'pkg_no': pd.Series(dtype='string')}).to_parquet(
                    Path(config['data_dir']) / 'test.parquet', index=False)
                if mutation == 'report':
                    changed = root / 'reports/splits/smoke_split.json'
                elif mutation == 'old_fit':
                    changed = root / 'artifacts/splits/smoke_split/dev_1/fit_ids.parquet'
                else:
                    changed = root / 'artifacts/splits/smoke_deploy_C/dev_1/cal_ids.parquet'
                changed.write_bytes(changed.read_bytes() + b'\n')
                if mutation == 'inherited_cal':
                    # Even a self-consistent new report cannot redefine a cohort
                    # that the protocol pins byte-for-byte to the old split.
                    report_path = root / 'reports/splits/smoke_deploy_C.json'
                    report = json.loads(report_path.read_text())
                    report['folds']['dev_1']['artifacts']['cal']['file_sha256'] = sha256_file(changed)
                    report_path.write_text(json.dumps(report))
                reads = []
                original = pd.read_parquet

                def reader(*args, **kwargs):
                    reads.append(kwargs.get('columns', []))
                    return original(*args, **kwargs)

                with patch.object(implementation.pd, 'read_parquet', side_effect=reader):
                    with self.assertRaisesRegex(ValueError, '[Rr]eference'):
                        implementation._prepare_inputs(config, 'dev_1', synthetic=True)
                    path = root / 'preflight.yaml'
                    path.write_text(yaml.safe_dump(config))
                    arguments = ['backtest.py', '--config', str(path), '--dry-run',
                                 '--run-id', 'SMOKE_REFERENCE_PREFLIGHT']
                    with patch.object(sys, 'argv', arguments), patch.object(cli, 'atomic_write_json') as write:
                        with self.assertRaisesRegex(ValueError, '[Rr]eference'):
                            cli.main()
                        write.assert_not_called()
                self.assertFalse(any('sign_time' in columns or 'y_total_hours' in columns for columns in reads))

    def test_cli_rejects_training_with_preparation_config(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'preparation.yaml'
            path.write_text(yaml.safe_dump({'experiment': 'B1', 'data_dir': '/does/not/exist',
                'execution': {'stage': 'preparation', 'allow_real_training': False}}))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/backtest.py'),
                                     '--config', str(path), '--execute-training',
                                     '--run-id', 'FORBIDDEN_SMOKE_CLI'],
                                    cwd=ROOT, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('TrainingNotAuthorizedError', result.stderr)
            self.assertFalse((ROOT / 'reports/runs/FORBIDDEN_SMOKE_CLI').exists())

    def test_real_training_is_blocked_before_data_access(self):
        config = {"experiment": "B1", "data_dir": "/does/not/exist",
                  "execution": {"stage": "preparation", "allow_real_training": False}}
        with self.assertRaises(TrainingNotAuthorizedError):
            run_backtest(config, "dev_1", "forbidden")
        with self.assertRaises(TrainingNotAuthorizedError):
            run_backtest(config, "dev_1", "forbidden", execute_training=True)
        config["execution"]["stage"] = "training"
        with self.assertRaises(TrainingNotAuthorizedError):
            run_backtest(config, "dev_1", "forbidden", execute_training=True)

    def test_synthetic_b1_end_to_end_and_no_overwrite(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root)
            result = run_backtest(config, "dev_1", "SMOKE_B1_dev_1", synthetic=True)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(result["synthetic"])
            self.assertTrue(result["not_competition_result"])
            self.assertEqual(result["metrics"], result["official_metrics"])
            prediction = pd.read_csv(root / "outputs/runs/SMOKE_B1_dev_1/predictions.csv", dtype={"pkg_no": "string"})
            self.assertEqual(prediction["pkg_no"].tolist(), ["SMOKE_0005", "SMOKE_0006"])
            self.assertTrue((prediction["pred_hours"] > 0).all())
            self.assertEqual(prediction["pred_hours"].tolist(), [25.0, 25.0])
            manifest = json.loads((root / "reports/runs/SMOKE_B1_dev_1/run_manifest.json").read_text())
            self.assertEqual(manifest['source_at_start']['snapshot_sha256'],
                             manifest['source_at_finish']['snapshot_sha256'])
            self.assertEqual(manifest['code_files_sha256']['models.py'],
                             manifest['source_at_start']['files_sha256']['src/f4/models.py'])
            self.assertIn("model", manifest["artifacts"])
            self.assertNotIn("calibrator", manifest["artifacts"])
            self.assertFalse((root / "outputs/runs/SMOKE_B1_dev_1/calibration_base_predictions.csv").exists())
            self.assertNotIn("y_total_hours", (root / "outputs/runs/SMOKE_B1_dev_1/predictions.csv").read_text())
            with self.assertRaises(FileExistsError):
                run_backtest(config, "dev_1", "SMOKE_B1_dev_1", synthetic=True)

    def test_smoke_flag_does_not_authorize_real_keys_or_large_models(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root)
            path = Path(config["data_dir"]) / "train.parquet"
            source = pd.read_parquet(path)
            source.loc[0, "pkg_no"] = "REAL_0000"
            source.to_parquet(path, index=False)
            with self.assertRaisesRegex(TrainingNotAuthorizedError, "SMOKE_"):
                run_backtest(config, "dev_1", "bad_keys", synthetic=True)
            large = pd.concat([source.iloc[[0]]] * 201, ignore_index=True)
            large["pkg_no"] = [f"SMOKE_{index:04d}" for index in range(len(large))]
            large.to_parquet(path, index=False)
            with self.assertRaisesRegex(TrainingNotAuthorizedError, "200 total input rows"):
                run_backtest(config, "dev_1", "large_input", synthetic=True)
            model_config = deepcopy(config)
            model_config.update(experiment="M1", model={"n_estimators": 6, "n_jobs": 1})
            with self.assertRaisesRegex(TrainingNotAuthorizedError, "five rounds"):
                run_backtest(model_config, "dev_1", "large_model", synthetic=True)
            model_config['model'] = {'n_estimators': 5, 'n_jobs': 1, 'num_iterations': 800}
            with self.assertRaisesRegex(ValueError, 'iteration aliases'):
                run_backtest(model_config, 'dev_1', 'alias_bypass', synthetic=True)
            with self.assertRaises(TrainingNotAuthorizedError):
                run_backtest(config, "holdout", "holdout_smoke", synthetic=True)

    def test_synthetic_m2_connects_encoder_model_and_independent_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root)
            config.update(experiment='M2', model={'n_estimators': 5, 'n_jobs': 1,
                                                 'min_child_samples': 1, 'num_leaves': 3},
                          calibration={'min_group_samples': 1})
            result = run_backtest(config, 'dev_1', 'SMOKE_M2_dev_1', synthetic=True)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(set(result['artifacts']), {'model', 'encoder', 'calibrator'})
            self.assertEqual(result['model_fit_ids_sha256'], result['calibration_fit_ids_sha256'])
            self.assertEqual(result['metrics'], result['official_metrics'])
            self.assertTrue((root / 'outputs/runs/SMOKE_M2_dev_1/calibration_base_predictions.csv').exists())

    def test_calibrated_baselines_share_m2_cohorts_and_translation_equivalence(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root, calibrated_baseline_fixture=True)
            results, predictions, bases = {}, {}, {}
            for experiment, kind in [('B0C', 'global_constant'), ('B1C', 'carrier_constant'), ('M2', None)]:
                candidate = deepcopy(config)
                candidate.update(experiment=experiment, calibration={'min_group_samples': 1})
                candidate['model'] = ({'kind': kind, 'min_group_samples': 1} if kind else
                                      {'n_estimators': 5, 'n_jobs': 1, 'min_child_samples': 1, 'num_leaves': 3})
                run_id = f'SMOKE_{experiment}_FAIR'
                result = run_backtest(candidate, 'dev_1', run_id, synthetic=True)
                results[experiment] = result
                predictions[experiment] = pd.read_csv(root / 'outputs/runs' / run_id / 'predictions.csv')
                bases[experiment] = pd.read_csv(root / 'outputs/runs' / run_id / 'calibration_base_predictions.csv')
                self.assertEqual(result['metrics'], result['official_metrics'])
                self.assertEqual(result['model_fit_ids_sha256'], result['calibration_fit_ids_sha256'])
                self.assertEqual(result['calibration_base_model_sha256'], result['artifacts']['model']['sha256'])
                self.assertIn('calibrator', result['artifacts'])
                self.assertEqual(bases[experiment]['pkg_no'].tolist(), ['SMOKE_0002', 'SMOKE_0003'])
            for result in results.values():
                self.assertEqual(result['source'], results['M2']['source'])
                self.assertEqual(result['cohort_n'], results['M2']['cohort_n'])
                self.assertEqual(result['calibration_ids_sha256'], results['M2']['calibration_ids_sha256'])
            self.assertNotEqual(bases['B0C']['pred_hours'].tolist(), bases['B1C']['pred_hours'].tolist())
            # Direct per-carrier calibration commitments: B=25h, A=73h.
            self.assertEqual(predictions['B0C']['pred_hours'].tolist(), [25.0, 73.0])
            pd.testing.assert_frame_equal(predictions['B0C'], predictions['B1C'])
            baseline = load_model(results['B1C']['artifacts']['model']['path'])
            self.assertEqual(baseline.predict(['B', 'A']).tolist(), [1.0, 25.0])

    def test_calibrated_baseline_still_rejects_fit_calibration_overlap(self):
        import f4.backtest as implementation
        original = implementation._prepare_inputs
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root, calibrated_baseline_fixture=True)
            config.update(experiment='B1C', calibration={'min_group_samples': 1})

            def overlapping_inputs(*args):
                provenance, cohorts, report = original(*args)
                cohorts['cal'] = cohorts['cal'].copy()
                cohorts['cal'].loc[0, 'pkg_no'] = cohorts['fit']['pkg_no'].iloc[0]
                return provenance, cohorts, report

            with patch.object(implementation, '_prepare_inputs', side_effect=overlapping_inputs):
                with self.assertRaisesRegex(ValueError, 'must not overlap'):
                    run_backtest(config, 'dev_1', 'SMOKE_B1C_OVERLAP', synthetic=True)
            manifest = json.loads((root / 'reports/runs/SMOKE_B1C_OVERLAP/run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertFalse((root / 'outputs/runs/SMOKE_B1C_OVERLAP/predictions.csv').exists())

    def test_cli_preflight_selects_correct_calibrated_baseline_kind(self):
        specification = importlib.util.spec_from_file_location('f4_backtest_cli_test', ROOT / 'scripts/backtest.py')
        cli = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cli)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root)
            for experiment, kind in [('B0C', 'global'), ('B1C', 'carrier')]:
                config['experiment'] = experiment
                config['model']['kind'] = kind + '_constant'
                path = root / f'{experiment}.yaml'
                path.write_text(yaml.safe_dump(config))
                arguments = ['backtest.py', '--config', str(path), '--dry-run', '--run-id', f'SMOKE_PREFLIGHT_{experiment}']
                with patch.object(sys, 'argv', arguments), patch('f4.models.QuantileBaseline') as baseline, \
                        patch.object(cli, 'atomic_write_json') as write, patch('builtins.print'):
                    # No test.parquet is supplied: preflight reports missing input without fitting.
                    self.assertEqual(cli.main(), 1)
                    baseline.assert_called_once_with(kind=kind, min_group_samples=1)
                    self.assertEqual(write.call_args.args[1]['experiment'], experiment)
                    baseline.return_value.fit.assert_not_called()
                    config['model']['kind'] = 'carrier_constant' if kind == 'global' else 'global_constant'
                    path.write_text(yaml.safe_dump(config))
                    with self.assertRaisesRegex(ValueError, 'kind disagrees'):
                        cli.main()

    def test_source_drift_after_scoring_invalidates_completed_backtest(self):
        import f4.backtest as implementation
        original_score = implementation.score_predictions
        baseline_hashes, _ = common.source_snapshot()
        changed = [False]

        def observed_source():
            hashes = dict(baseline_hashes)
            if changed[0]:
                hashes['src/f4/models.py'] = '0' * 64
            return hashes, common.configuration_hash(hashes)

        def score_then_drift(*args, **kwargs):
            result = original_score(*args, **kwargs)
            changed[0] = True
            return result

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_project(root)
            with patch.object(common, 'source_snapshot', side_effect=observed_source), \
                    patch.object(implementation, 'score_predictions', side_effect=score_then_drift):
                with self.assertRaises(common.ExecutionInvalidatedError):
                    run_backtest(config, 'dev_1', 'SMOKE_SOURCE_DRIFT', synthetic=True)
            manifest = json.loads((root / 'reports/runs/SMOKE_SOURCE_DRIFT/run_manifest.json').read_text())
            self.assertEqual(manifest['status'], 'invalidated')
            self.assertEqual(manifest['failure_kind'], 'execution_snapshot_changed')
            self.assertNotEqual(manifest['source_at_start']['snapshot_sha256'],
                                manifest['source_at_finish']['snapshot_sha256'])


if __name__ == "__main__":
    unittest.main()

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

from f4.backtest import run_backtest
from f4.features import SHARED_COLUMNS
from f4.models import load_model
from f4.safety import TrainingNotAuthorizedError
from f4.splits import make_splits


def synthetic_project(root, *, calibrated_baseline_fixture=False):
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
    source = pd.DataFrame("KNOWN", index=range(len(rows)), columns=SHARED_COLUMNS)
    source["pkg_no"] = [f"SMOKE_{index:04d}" for index in range(len(rows))]
    source["pkg_goods_cnt"], source["weight"] = 1.0, 0.5
    source["issue_time"] = pd.to_datetime([row[0] for row in rows])
    source["sign_time"] = pd.to_datetime([row[1] for row in rows])
    source["pay_time"] = source["issue_time"] - pd.Timedelta(hours=1)
    source["y_total_hours"] = (source["sign_time"] - source["pay_time"]).dt.total_seconds() / 3600
    source["lgs_company_nm"] = ["A", "B"] * 5
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


class BacktestSmokeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

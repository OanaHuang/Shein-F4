"""Only synthetic end-to-end backtests and preparation authorization checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import subprocess

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f4.backtest import run_backtest
from f4.features import SHARED_COLUMNS
from f4.safety import TrainingNotAuthorizedError
from f4.splits import make_splits


def synthetic_project(root):
    rows = [
        ("2026-08-01", "2026-08-02"), ("2026-08-02", "2026-08-03"),
        ("2026-08-03", "2026-08-06"), ("2026-08-04", "2026-08-08"),
        ("2026-08-05", "2026-08-12"), ("2026-08-08", "2026-08-30"),
        ("2026-08-09", "2026-08-11"), ("2026-08-12", "2026-08-15"),
        ("2026-08-13", "2026-08-16"), ("2026-08-14", "2026-08-17"),
    ]
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
            manifest = json.loads((root / "reports/runs/SMOKE_B1_dev_1/run_manifest.json").read_text())
            self.assertIn("model", manifest["artifacts"])
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


if __name__ == "__main__":
    unittest.main()

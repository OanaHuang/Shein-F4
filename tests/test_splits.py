"""Synthetic checks for maturity boundaries and final-holdout isolation."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f4.splits import FoldSpec, build_fold, holdout_keys, make_splits, normalize_features, validate_config


def fixture():
    rows = [
        ("0001", "2026-08-01", "2026-08-02"),
        ("0002", "2026-08-02", "2026-08-03"),
        ("0003", "2026-08-03", "2026-08-06"),
        ("0004", "2026-08-04", "2026-08-08"),
        ("0005", "2026-08-05", "2026-08-12"),
        ("0006", "2026-08-08", "2026-08-30"),
        ("0007", "2026-08-09", "2026-08-11"),
        ("0008", "2026-08-12", "2026-08-15"),
        ("0009", "2026-08-13", "2026-08-16"),
        ("0010", "2026-08-14", "2026-08-17"),
    ]
    frame = pd.DataFrame(rows, columns=["pkg_no", "issue_time", "sign_time"])
    frame["pay_time"] = pd.to_datetime(frame["issue_time"]) - pd.Timedelta(hours=1)
    frame["lgs_company_nm"] = ["A", "B"] * 5
    frame["y_total_hours"] = 30.0
    config = {"split_id": "synthetic_v1",
              "holdout": {"F": "2026-08-04", "E": "2026-08-06", "C": "2026-08-12", "H_days": 2},
              "development": [{"name": "dev_1", "F": "2026-08-03", "E": "2026-08-05", "C": "2026-08-08", "H_days": 2}]}
    return frame, config


class SplitSmokeTests(unittest.TestCase):
    def test_maturity_and_eval_boundaries(self):
        frame, config = fixture()
        _, final, development = validate_config(config)
        features = normalize_features(frame)
        frozen = holdout_keys(features, final)
        keys, report = build_fold(features, frame, development[0], frozen)
        self.assertEqual(keys["fit"]["pkg_no"].tolist(), ["0001"])
        # sign_time == C is excluded; issue_time == E is outside the cal queue.
        self.assertEqual(keys["cal"]["pkg_no"].tolist(), ["0003"])
        # A slow evaluation row signed much later is retained.
        self.assertEqual(keys["eval"]["pkg_no"].tolist(), ["0006", "0007"])
        self.assertEqual(frozen["pkg_no"].tolist(), ["0008", "0009"])
        self.assertEqual(report["calibration_maturity"]["queue_n"], 2)
        self.assertEqual(report["calibration_maturity"]["mature_fraction"], .5)
        self.assertNotIn("max_used_sign_time", report["segments"]["eval"])

    def test_cross_window_holdout_overlap_is_rejected(self):
        frame, config = fixture()
        features = normalize_features(frame)
        _, final, _ = validate_config(config)
        frozen = holdout_keys(features, final)
        bad = FoldSpec.from_mapping({"name": "dev_bad", "F": "2026-08-03", "E": "2026-08-05", "C": "2026-08-09", "H_days": 4})
        with self.assertRaisesRegex(ValueError, "overlaps final holdout"):
            build_fold(features, frame, bad, frozen)

    def test_holdout_label_changes_cannot_change_development_keys(self):
        frame, config = fixture()
        features = normalize_features(frame)
        _, final, development = validate_config(config)
        frozen = holdout_keys(features, final)
        keys, report = build_fold(features, frame, development[0], frozen)
        changed = frame.copy()
        mask = changed["pkg_no"].isin(frozen["pkg_no"])
        changed.loc[mask, "sign_time"] = "not a timestamp"
        changed.loc[mask, "y_total_hours"] = -9999
        new_keys, new_report = build_fold(features, changed, development[0], frozen)
        for phase in keys:
            pd.testing.assert_frame_equal(keys[phase], new_keys[phase])
        self.assertEqual(report, new_report)
        pd.testing.assert_frame_equal(frozen, holdout_keys(normalize_features(changed), final))

    def test_freeze_is_idempotent_and_rejects_new_provenance(self):
        frame, config = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / "train.parquet", index=False)
            path = root / "splits.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            report_path = root / "report.json"
            original_read = pd.read_parquet
            reads = []
            def tracked_read(*args, **kwargs):
                reads.append(kwargs.get("columns"))
                if "sign_time" in (kwargs.get("columns") or []):
                    self.assertTrue((root / "artifacts/synthetic_v1/holdout_ids.parquet").exists())
                    self.assertEqual(json.loads(report_path.read_text())["status"], "holdout_locked")
                return original_read(*args, **kwargs)
            with patch("f4.splits.pd.read_parquet", side_effect=tracked_read):
                first = make_splits(root, path, root / "artifacts", report_path)
            self.assertEqual(reads[0], ["pkg_no", "pay_time", "issue_time"])
            self.assertFalse(any("y_total_hours" in (columns or []) for columns in reads))
            second = make_splits(root, path, root / "artifacts", report_path)
            self.assertEqual(first, second)
            self.assertTrue(first["holdout_isolation_checked"])
            self.assertEqual(first["status"], "complete")
            self.assertEqual(pd.read_parquet(root / "artifacts/synthetic_v1/holdout_ids.parquet")["pkg_no"].tolist(), ["0008", "0009"])
            config["development"][0]["E"] = "2026-08-06"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                make_splits(root, path, root / "artifacts", report_path)
            self.assertEqual(json.loads(report_path.read_text()), first)

    def test_invalid_identifiers_and_duplicate_keys_are_rejected(self):
        frame, config = fixture()
        config["split_id"] = "../escape"
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_features(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()

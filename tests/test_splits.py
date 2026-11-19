"""Synthetic checks for maturity boundaries and final-holdout isolation."""

from copy import deepcopy
import hashlib
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

from f4.splits import (FoldSpec, build_fold, canonical_split_config, holdout_keys,
                       make_splits, normalize_features, validate_config)


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


def deploy_c_config(config, reference_report, reference_root):
    result = deepcopy(config)
    result.update(split_id='synthetic_deploy_c', protocol='asof_deploy_C_v1',
                  reference={'split_id': config['split_id'], 'report': str(reference_report),
                             'report_sha256': hashlib.sha256(reference_report.read_bytes()).hexdigest(),
                             'artifacts': str(reference_root)})
    for value in [result['holdout'], *result['development']]:
        for field in ['model_fit_time', 'fit_label_cutoff', 'cal_label_cutoff']:
            value[field] = value['C']
    return result


class SplitSmokeTests(unittest.TestCase):
    def test_legacy_canonical_metadata_has_exact_original_shape(self):
        _, config = fixture()
        expected = {
            'split_id': 'synthetic_v1',
            'holdout': {'name': 'holdout', 'F': '2026-08-04T00:00:00',
                        'E': '2026-08-06T00:00:00', 'C': '2026-08-12T00:00:00',
                        'H_days': 2, 'eval_end': '2026-08-14T00:00:00'},
            'development': [{'name': 'dev_1', 'F': '2026-08-03T00:00:00',
                             'E': '2026-08-05T00:00:00', 'C': '2026-08-08T00:00:00',
                             'H_days': 2, 'eval_end': '2026-08-10T00:00:00'}],
        }
        self.assertEqual(canonical_split_config(config), expected)
        config['protocol'] = 'asof_backtest'
        self.assertEqual(canonical_split_config(config), expected)

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
                    requested = set(kwargs['filters'][0][2])
                    self.assertFalse(requested.intersection({'0008', '0009'}))
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

    def test_deploy_c_expands_only_fit_and_preserves_reference_artifact_bytes(self):
        frame, config = fixture()
        # A pre-F parcel whose sign is exactly C must stay excluded in both protocols.
        extra = frame.iloc[[1]].copy()
        extra['pkg_no'], extra['sign_time'] = '0011', '2026-08-08'
        frame = pd.concat([frame, extra], ignore_index=True)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / 'train.parquet', index=False)
            old_path, old_report = root / 'legacy.yaml', root / 'legacy.json'
            old_path.write_text(yaml.safe_dump(config))
            old = make_splits(root, old_path, root / 'artifacts', old_report)
            old_root = root / 'artifacts' / config['split_id']
            old_files = {p: p.read_bytes() for p in old_root.rglob('*.parquet')}
            old_report_bytes = old_report.read_bytes()
            current_config = deploy_c_config(config, old_report, old_root)
            current_path = root / 'deploy_c.yaml'
            current_path.write_text(yaml.safe_dump(current_config))
            new = make_splits(root, current_path, root / 'artifacts', root / 'deploy_c.json')
            new_root = root / 'artifacts' / current_config['split_id']
            old_fit = pd.read_parquet(old_root / 'dev_1/fit_ids.parquet')['pkg_no'].tolist()
            new_fit = pd.read_parquet(new_root / 'dev_1/fit_ids.parquet')['pkg_no'].tolist()
            self.assertEqual(old_fit, ['0001'])
            self.assertEqual(new_fit, ['0001', '0002'])
            self.assertTrue(set(old_fit).issubset(new_fit))
            self.assertNotIn('0011', new_fit)
            self.assertNotIn('0003', new_fit)  # issue == F belongs only to calibration.
            self.assertEqual(new['folds']['dev_1']['protocol'], 'asof_deploy_C_v1')
            self.assertEqual(new['folds']['dev_1']['times']['model_fit_time'], '2026-08-08T00:00:00')
            self.assertEqual(new['folds']['dev_1']['times']['fit_label_cutoff'], '2026-08-08T00:00:00')
            self.assertEqual(new['folds']['dev_1']['times']['cal_label_cutoff'], '2026-08-08T00:00:00')
            self.assertEqual(new['holdout']['file_sha256'], old['holdout']['file_sha256'])
            self.assertEqual((new_root / 'holdout_ids.parquet').read_bytes(), old_files[old_root / 'holdout_ids.parquet'])
            for fold in ['dev_1', 'holdout']:
                for phase in ['cal', 'eval']:
                    relative = Path(fold) / f'{phase}_ids.parquet'
                    self.assertEqual((new_root / relative).read_bytes(), old_files[old_root / relative])
                    self.assertEqual(new['folds'][fold]['artifacts'][phase]['file_sha256'],
                                     old['folds'][fold]['artifacts'][phase]['file_sha256'])
            self.assertEqual(pd.read_parquet(new_root / 'dev_1/cal_ids.parquet')['pkg_no'].tolist(), ['0003'])
            self.assertEqual(pd.read_parquet(new_root / 'dev_1/eval_ids.parquet')['pkg_no'].tolist(), ['0006', '0007'])
            self.assertEqual(old_report.read_bytes(), old_report_bytes)
            for path, original in old_files.items():
                self.assertEqual(path.read_bytes(), original)

    def test_deploy_c_never_reads_isolated_sign_values(self):
        frame, config = fixture()
        isolated = {'0008', '0009'}
        frame.loc[frame['pkg_no'].isin(isolated), 'sign_time'] = 'HOLDOUT_POISON'
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / 'train.parquet', index=False)
            old_path, old_report = root / 'legacy.yaml', root / 'legacy.json'
            old_path.write_text(yaml.safe_dump(config))
            make_splits(root, old_path, root / 'artifacts', old_report)
            new_config = deploy_c_config(config, old_report, root / 'artifacts' / config['split_id'])
            new_path = root / 'deploy_c.yaml'
            new_path.write_text(yaml.safe_dump(new_config))
            original_read = pd.read_parquet
            sign_reads = []

            def tracked_read(*args, **kwargs):
                if 'sign_time' in kwargs.get('columns', []):
                    selected = set(kwargs['filters'][0][2])
                    self.assertFalse(selected.intersection(isolated))
                    sign_reads.append(selected)
                value = original_read(*args, **kwargs)
                if 'sign_time' in kwargs.get('columns', []):
                    self.assertFalse(set(value['pkg_no']).intersection(isolated))
                return value

            with patch('f4.splits.pd.read_parquet', side_effect=tracked_read):
                result = make_splits(root, new_path, root / 'artifacts', root / 'deploy_c.json')
            self.assertTrue(sign_reads)
            self.assertEqual(result['status'], 'complete')

    def test_bad_protocol_or_cutoffs_fail_before_any_parquet_read(self):
        frame, config = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / 'train.parquet', index=False)
            old_path, old_report = root / 'legacy.yaml', root / 'legacy.json'
            old_path.write_text(yaml.safe_dump(config))
            make_splits(root, old_path, root / 'artifacts', old_report)
            valid = deploy_c_config(config, old_report, root / 'artifacts' / config['split_id'])
            cases = []
            bad = deepcopy(valid); bad['protocol'] = 'unrecognized_protocol'; cases.append(bad)
            bad = deepcopy(valid); del bad['development'][0]['fit_label_cutoff']; cases.append(bad)
            bad = deepcopy(valid); bad['development'][0]['fit_label_cutoff'] = '2026-08-08 00:00:01'; cases.append(bad)
            bad = deepcopy(valid); bad['development'][0]['cal_label_cutoff'] = '2026-08-09'; cases.append(bad)
            bad = deepcopy(valid); bad['development'][0]['model_fit_time'] = '2026-08-09'; cases.append(bad)
            for index, bad in enumerate(cases):
                path = root / f'bad_{index}.yaml'
                path.write_text(yaml.safe_dump(bad))
                with self.subTest(index=index), patch('f4.splits.pd.read_parquet') as read:
                    with self.assertRaises(ValueError):
                        make_splits(root, path, root / 'artifacts', root / f'bad_{index}.json')
                    read.assert_not_called()

    def test_changed_reference_key_is_rejected_before_sign_read(self):
        frame, config = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / 'train.parquet', index=False)
            old_path, old_report = root / 'legacy.yaml', root / 'legacy.json'
            old_path.write_text(yaml.safe_dump(config))
            make_splits(root, old_path, root / 'artifacts', old_report)
            old_root = root / 'artifacts' / config['split_id']
            new_config = deploy_c_config(config, old_report, old_root)
            path = root / 'deploy_c.yaml'
            path.write_text(yaml.safe_dump(new_config))
            pd.DataFrame({'pkg_no': ['0004']}).to_parquet(old_root / 'dev_1/cal_ids.parquet', index=False)
            original_read = pd.read_parquet

            def tracked_read(*args, **kwargs):
                self.assertNotIn('sign_time', kwargs.get('columns', []))
                return original_read(*args, **kwargs)

            with patch('f4.splits.pd.read_parquet', side_effect=tracked_read):
                with self.assertRaises(ValueError):
                    make_splits(root, path, root / 'artifacts', root / 'deploy_c.json')

    def test_changed_reference_fit_is_rejected_before_sign_read(self):
        frame, config = fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_parquet(root / 'train.parquet', index=False)
            old_path, old_report = root / 'legacy.yaml', root / 'legacy.json'
            old_path.write_text(yaml.safe_dump(config))
            make_splits(root, old_path, root / 'artifacts', old_report)
            old_root = root / 'artifacts' / config['split_id']
            path = root / 'deploy_c.yaml'
            path.write_text(yaml.safe_dump(deploy_c_config(config, old_report, old_root)))
            # A damaged old fit cannot be checked only after new labels are read.
            pd.DataFrame({'pkg_no': ['0004']}).to_parquet(old_root / 'dev_1/fit_ids.parquet', index=False)
            original_read = pd.read_parquet
            reads = []

            def tracked_read(*args, **kwargs):
                reads.append(kwargs.get('columns', []))
                return original_read(*args, **kwargs)

            with patch('f4.splits.pd.read_parquet', side_effect=tracked_read):
                with self.assertRaises(ValueError):
                    make_splits(root, path, root / 'artifacts', root / 'deploy_c.json')
            self.assertFalse(any('sign_time' in columns for columns in reads),
                             'Corrupted reference fit was detected after reading sign labels')

    def test_invalid_identifiers_and_duplicate_keys_are_rejected(self):
        frame, config = fixture()
        config["split_id"] = "../escape"
        with self.assertRaises(ValueError):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, "unique"):
            normalize_features(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))


if __name__ == "__main__":
    unittest.main()

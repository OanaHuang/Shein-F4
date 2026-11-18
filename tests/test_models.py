"""Synthetic-only feature/model/calibration smoke checks; no F4 data is read."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from f4.calibration import ResidualCalibrator
from f4.features import FeatureBuilder, SHARED_COLUMNS
from f4.models import (ConditionalQuantileModel, QuantileBaseline,
                       empirical_quantile, load_model, save_model)


def synthetic_raw(n=8, prefix=''):
    """Issue-time fields only; keys deliberately contain leading zeroes."""
    frame = pd.DataFrame({column: [column + '_A'] * n for column in SHARED_COLUMNS})
    frame['pkg_no'] = [f'{prefix}{index:06d}' for index in range(n)]
    frame['weight'] = np.arange(n, dtype='float32') * 10 + 200
    frame['pkg_goods_cnt'] = np.arange(n) % 4 + 1
    frame['pay_time'] = pd.date_range('2026-08-01', periods=n, freq='h')
    frame['issue_time'] = frame['pay_time'] + pd.Timedelta(hours=3)
    frame.index = np.arange(n) * 2 + 100
    return frame


class FeatureSmoke(unittest.TestCase):
    def test_unknown_categories_and_dictionary_roundtrip(self):
        raw = synthetic_raw()
        builder = FeatureBuilder().fit(raw)
        changed = synthetic_raw(3, prefix='eval_')
        changed.loc[changed.index[0], 'shpp_chnl'] = 'NEW_CHANNEL'
        changed.loc[changed.index[1], 'shpp_chnl'] = None
        result = builder.transform(changed)
        self.assertEqual(list(result.index), list(changed.index))
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['unknown__shpp_chnl'], 1)
        self.assertEqual(result.iloc[0]['cat__shpp_chnl'], 1)
        self.assertEqual(result.iloc[1]['missing__shpp_chnl'], 1)
        self.assertEqual(result.iloc[1]['cat__shpp_chnl'], 0)
        self.assertNotIn('NEW_CHANNEL', builder.category_maps_['shpp_chnl'])
        self.assertNotIn('pkg_no', result)
        self.assertFalse(any('post_no' in column for column in result))
        self.assertEqual(raw.iloc[0]['pkg_no'], '000000')
        restored = FeatureBuilder.from_dict(json.loads(json.dumps(builder.to_dict())))
        pd.testing.assert_frame_equal(result, restored.transform(changed))
        tampered = json.loads(json.dumps(builder.to_dict()))
        tampered['category_maps']['shpp_chnl']['shpp_chnl_A'] = 99
        with self.assertRaisesRegex(ValueError, 'digest'):
            FeatureBuilder.from_dict(tampered)

    def test_extra_targets_nodes_and_bad_keys_are_rejected(self):
        for column in ('y_total_hours', 'sign_time', 'pack_time', 'future_unknown'):
            raw = synthetic_raw()
            raw[column] = 1
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, 'Forbidden'):
                FeatureBuilder().fit(raw)
        raw = synthetic_raw()
        raw.loc[raw.index[1], 'pkg_no'] = raw.iloc[0]['pkg_no']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            FeatureBuilder().fit(raw)
        raw = synthetic_raw().drop(columns='issue_time')
        with self.assertRaisesRegex(ValueError, 'Missing shared'):
            FeatureBuilder().fit(raw)

    def test_invalid_weights_counts_and_negative_elapsed_are_flagged(self):
        raw = synthetic_raw()
        raw.loc[raw.index[0], 'pkg_goods_cnt'] = 0
        raw.loc[raw.index[1], 'weight'] = np.inf
        raw.loc[raw.index[2], 'issue_time'] = raw.iloc[2]['pay_time'] - pd.Timedelta(hours=1)
        result = FeatureBuilder().fit_transform(raw)
        self.assertTrue(np.isnan(result.iloc[0]['weight_per_item']))
        self.assertEqual(result.iloc[0]['goods_count_invalid'], 1)
        self.assertTrue(np.isnan(result.iloc[1]['weight']))
        self.assertEqual(result.iloc[1]['weight_invalid'], 1)
        self.assertEqual(result.iloc[2]['elapsed_negative'], 1)
        self.assertTrue(np.isnan(result.iloc[2]['elapsed_hours']))
        self.assertFalse(np.isinf(result.to_numpy()).any())


class StatisticalModelSmoke(unittest.TestCase):
    def test_order_statistics_and_carrier_fallback(self):
        self.assertEqual(empirical_quantile(np.arange(1, 11), .90), 9)
        self.assertEqual(empirical_quantile(np.arange(1, 11), .97), 10)
        target = np.array([10.] * 10 + [100.])
        groups = ['A'] * 10 + ['B']
        model = QuantileBaseline('carrier', min_group_samples=3).fit(
            target, groups, pkg_ids=[f'fit_{i:03d}' for i in range(11)])
        predicted, detail = model.predict(['A', 'B', 'NEW', None], return_diagnostics=True)
        np.testing.assert_allclose(predicted, [10, 76, 76, 76])
        self.assertEqual(detail['fallback_level'].tolist(), ['carrier', 'global', 'global', 'global'])
        self.assertEqual(detail['fitting_n'].tolist(), [10, 1, 0, 0])
        global_model = QuantileBaseline().fit(target, pkg_ids=[f'fit_{i}' for i in range(11)])
        np.testing.assert_allclose(global_model.predict(n_rows=2), [76, 76])

    def test_baseline_rejects_invalid_labels_and_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, 'finite'):
            QuantileBaseline().fit([10, np.inf], pkg_ids=['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'positive'):
            QuantileBaseline().fit([0, 10], pkg_ids=['a', 'b'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            QuantileBaseline().fit([10, 20], pkg_ids=['a', 'a'])

    def test_independent_residual_coverage_and_sparse_fallback(self):
        residual = np.array([0.] * 90 + [20.] * 7 + [50.] * 3)
        base = np.full(100, 100.)
        target = base + residual
        calibrator = ResidualCalibrator(min_group_samples=100).fit(
            target, base, ['A'] * 100, fit_ids=['fit_000'],
            cal_ids=[f'cal_{i:03d}' for i in range(100)])
        predicted = calibrator.apply(base, ['A'] * 100)
        self.assertGreaterEqual(float(np.mean(target <= predicted)), .90)
        self.assertLessEqual(float(np.mean(target - predicted > 24)), .03)
        _, detail = calibrator.apply([100.], ['NEW'], return_diagnostics=True)
        self.assertEqual(detail.iloc[0]['fallback_level'], 'global')

    def test_calibration_overlap_and_wrong_model_digest_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            ResidualCalibrator().fit([10.], [9.], ['A'], fit_ids=['same'], cal_ids=['same'])
        with self.assertRaisesRegex(ValueError, 'digest'):
            ResidualCalibrator().fit([10.], [9.], ['A'], fit_ids=['fit'],
                                     cal_ids=['cal'], model_fit_ids_sha256='wrong')

    def test_negative_calibration_and_output_floor(self):
        calibrator = ResidualCalibrator(min_group_samples=1).fit(
            [10., 10., 10.], [100., 100., 100.], ['A'] * 3,
            fit_ids=['fit'], cal_ids=['c1', 'c2', 'c3'])
        self.assertEqual(calibrator.global_delta_, -90)
        predicted = calibrator.apply([100., .5], ['A', 'A'])
        self.assertEqual(predicted[0], 10)
        self.assertGreater(predicted[1], 0)


class ConditionalModelSmoke(unittest.TestCase):
    def test_synthetic_five_round_single_thread_fit_predict_and_artifact(self):
        # This is the only LightGBM fit in this file. It uses no real package data.
        raw = synthetic_raw(120)
        builder = FeatureBuilder()
        matrix = builder.fit_transform(raw)
        target = pd.Series(30 + np.arange(120) / 10, index=raw.index)
        model = ConditionalQuantileModel(params={
            'n_estimators': 5, 'n_jobs': 1, 'num_leaves': 7,
            'min_child_samples': 5,
        }).fit(matrix, target, pkg_ids=raw['pkg_no'])
        evaluation = synthetic_raw(10, prefix='eval_')
        evaluation.loc[evaluation.index[0], 'shpp_chnl'] = 'NEW'
        transformed = builder.transform(evaluation)
        quantiles = model.predict_quantiles(transformed)
        predicted = model.predict(transformed)
        self.assertTrue(np.isfinite(predicted).all())
        self.assertTrue((predicted > 0).all())
        self.assertTrue((quantiles['q97'] >= quantiles['q90']).all())
        self.assertEqual(list(quantiles.index), list(evaluation.index))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.pkl'
            save_model(model, path)
            np.testing.assert_allclose(load_model(path).predict(transformed), predicted)
            with self.assertRaises(FileExistsError):
                save_model(model, path)
        wrong_encoder = transformed.copy()
        wrong_encoder.attrs['feature_fit_ids_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'encoder'):
            model.predict(wrong_encoder)
        wrong_dictionary = transformed.copy()
        wrong_dictionary.attrs['feature_encoder_sha256'] = 'changed dictionary'
        with self.assertRaisesRegex(ValueError, 'dictionary'):
            model.predict(wrong_dictionary)
        with self.assertRaisesRegex(ValueError, 'schema/order'):
            model.predict(transformed.iloc[:, ::-1])
        with self.assertRaisesRegex(ValueError, 'bound'):
            ConditionalQuantileModel().fit(matrix, target, pkg_ids=raw['pkg_no'].tolist()[::-1])
        forbidden = matrix.copy()
        forbidden['sign_time'] = 1
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            ConditionalQuantileModel().fit(forbidden, target, pkg_ids=raw['pkg_no'])


class ParameterGuardSmoke(unittest.TestCase):
    def test_iteration_objective_and_thread_aliases_cannot_bypass_controls(self):
        for key in ['max_iter', 'n_iter', 'nrounds', 'num_boost_round', 'num_iteration',
                    'num_iterations', 'num_round', 'num_rounds', 'num_tree', 'num_trees']:
            with self.subTest(alias=key), self.assertRaisesRegex(ValueError, 'iteration aliases'):
                ConditionalQuantileModel({'n_estimators': 5, key: 800})
        for key in ['loss', 'application', 'objective_type', 'app']:
            with self.subTest(alias=key), self.assertRaisesRegex(ValueError, 'Objective aliases'):
                ConditionalQuantileModel({key: 'regression'})
        for key in ['num_threads', 'nthread', 'num_thread', 'nthreads']:
            with self.subTest(alias=key), self.assertRaisesRegex(ValueError, 'n_jobs'):
                ConditionalQuantileModel({'n_jobs': 1, key: 8})


if __name__ == '__main__':
    unittest.main()

import importlib.util
from io import StringIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from f4.metrics import read_prediction_csv, score_predictions, validate_predictions
from f4.safety import require_preparation_mode, require_synthetic, TrainingNotAuthorizedError


class ScoreTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'official/tools/score.py'
        spec = importlib.util.spec_from_file_location('official_score', path)
        self.official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.official)

    def test_official_boundaries_and_alignment(self):
        truth = pd.DataFrame({'pkg_no': [f'SMOKE_{i:03}' for i in range(200)],
                              'lgs_company_nm': ['A'] * 100 + ['B'] * 100,
                              'y_total_hours': [100.] * 200})
        T = np.array([100.] * 90 + [76.] * 7 + [75.999] * 3
                     + [99.] * 20 + [100.] * 80)
        pred = pd.DataFrame({'pkg_no': truth.pkg_no, 'pred_hours': T})
        score, groups = score_predictions(truth, pred.sample(frac=1, random_state=7))
        expected = self.official.score(truth.lgs_company_nm, truth.y_total_hours, T)
        for key in expected:
            self.assertAlmostEqual(score[key], expected[key], places=10)
        first = groups.set_index('group').loc['A']
        self.assertEqual(first.ontime_rate, .90)
        self.assertEqual(first.severe_rate, .03)
        self.assertEqual(first.pen_ontime, 0)
        self.assertEqual(first.pen_severe, 0)
        self.assertAlmostEqual(groups.pen_ontime.sum(), score['pen_ontime'])

    def test_reject_invalid_keys_and_values(self):
        items = pd.DataFrame({'pkg_no': ['001', '002']})
        valid = pd.DataFrame({'pkg_no': ['002', '001'], 'pred_hours': [2., 1.]})
        self.assertEqual(validate_predictions(items, valid).pred_hours.tolist(), [1., 2.])
        arrow_items = items.astype({'pkg_no': 'string[pyarrow]'})
        self.assertEqual(validate_predictions(arrow_items, valid).pred_hours.tolist(), [1., 2.])
        bad_frames = [valid.assign(pkg_no=['001', '001']), valid.iloc[:1],
                      valid.assign(pkg_no=['001', '003']), valid.assign(extra=1)]
        for number in [0, -1, np.nan, np.inf, 'bad']:
            broken = valid.astype({'pred_hours': object}).copy()
            broken.loc[0, 'pred_hours'] = number
            bad_frames.append(broken)
        for broken in bad_frames:
            with self.subTest(frame=str(broken)):
                with self.assertRaises(ValueError): validate_predictions(items, broken)

    def test_saved_prediction_keeps_float_at_scoring_boundary_and_literal_keys(self):
        hours = 1298621 / 3600
        predictions = read_prediction_csv(StringIO(f"pkg_no,pred_hours\n001,{hours}\nNA,{hours}\n"))
        self.assertEqual(predictions["pkg_no"].tolist(), ["001", "NA"])
        self.assertEqual(predictions["pred_hours"].tolist(), [hours, hours])
        truth = pd.DataFrame({"pkg_no": ["001", "NA"], "lgs_company_nm": ["A", "A"],
                              "y_total_hours": [hours, hours]})
        self.assertEqual(score_predictions(truth, predictions)[0]["A"], 0)

    def test_small_groups_are_not_weighted(self):
        truth = pd.DataFrame({'pkg_no': ['SMOKE_0', 'SMOKE_1', 'SMOKE_2'],
                              'lgs_company_nm': ['A', 'B', 'B'], 'y_total_hours': [100.] * 3})
        pred = pd.DataFrame({'pkg_no': truth.pkg_no, 'pred_hours': [1.] * 3})
        score, _ = score_predictions(truth, pred)
        self.assertAlmostEqual(score['pen_ontime'], 900)
        self.assertAlmostEqual(score['pen_severe'], 970)

    def test_preparation_guard(self):
        with self.assertRaises(TrainingNotAuthorizedError): require_preparation_mode(False)
        with self.assertRaises(TrainingNotAuthorizedError):
            require_synthetic(pd.DataFrame({'pkg_no': ['PKG_REAL']}))
        require_synthetic(pd.DataFrame({'pkg_no': ['SMOKE_001']}))


if __name__ == '__main__':
    unittest.main()

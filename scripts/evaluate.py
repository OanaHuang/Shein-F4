"""Key-aligned score CLI; preparation stage permits synthetic smoke data only."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from f4.common import atomic_write_json, project_path
from f4.metrics import score_predictions
from f4.safety import require_preparation_mode, require_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', required=True)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--out', default='reports/smoke/evaluation')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    require_preparation_mode(args.synthetic)
    truth = pd.read_csv(project_path(args.truth), dtype={'pkg_no': 'string'})
    pred = pd.read_csv(project_path(args.pred), dtype={'pkg_no': 'string'})
    require_synthetic(truth)
    metrics, groups = score_predictions(truth, pred)
    out = project_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / 'metrics.json', {'data_kind': 'synthetic', 'metrics': metrics})
    groups.to_csv(out / 'by_carrier.csv', index=False)
    print(metrics)


if __name__ == '__main__':
    main()

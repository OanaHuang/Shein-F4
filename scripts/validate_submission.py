"""Validate package keys and finite positive hour predictions without labels."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
from f4.common import atomic_write_json, project_path, sha256_file, timestamp
from f4.metrics import validate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred', required=True)
    parser.add_argument('--items', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    pred_path, items_path = project_path(args.pred), project_path(args.items)
    try:
        if items_path.suffix == '.parquet':
            items = pd.read_parquet(items_path, columns=['pkg_no'])
        else:
            items = pd.read_csv(items_path, usecols=['pkg_no'], dtype={'pkg_no': 'string'})
        pred = pd.read_csv(pred_path, dtype={'pkg_no': 'string'})
        aligned = validate_predictions(items, pred)
    except (ValueError, KeyError, OSError) as exc:
        atomic_write_json(project_path(args.report), {
            'checked_at': timestamp(), 'valid': False,
            'error': f'{type(exc).__name__}: {exc}',
        })
        print(f'Invalid submission: {exc}', file=sys.stderr)
        return 1
    atomic_write_json(project_path(args.report), {
        'checked_at': timestamp(), 'valid': True, 'rows': len(aligned),
        'prediction_sha256': sha256_file(pred_path), 'items_sha256': sha256_file(items_path),
        'min_pred_hours': float(aligned['pred_hours'].min()),
        'max_pred_hours': float(aligned['pred_hours'].max()),
        'quantiles': aligned['pred_hours'].quantile([.01, .5, .9, .99]).to_dict(),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare core tables without fitting or scoring a model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from f4.data import EXPECTED_ROWS, prepare_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', default='data/raw/official')
    parser.add_argument('--out', default='data/processed/core')
    parser.add_argument('--chunk-size', type=int, default=200_000)
    parser.add_argument('--expected-train-rows', type=int, default=EXPECTED_ROWS['train'])
    parser.add_argument('--expected-test-rows', type=int, default=EXPECTED_ROWS['test'])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    raw = ROOT / args.raw_root
    out = ROOT / args.out
    result = prepare_data(raw, out, chunk_size=args.chunk_size, overwrite=args.overwrite,
                          expected_rows={'train': args.expected_train_rows, 'test': args.expected_test_rows})
    print(f"Prepared {result['files']['train']['rows']:,} train and {result['files']['test']['rows']:,} test rows")


if __name__ == '__main__':
    main()

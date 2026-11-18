#!/usr/bin/env python3
"""Audit shared features, or development labels after excluding frozen holdout IDs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from f4.data import audit_development, audit_structure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['structure', 'development'], required=True)
    parser.add_argument('--data-dir', default='data/processed/core')
    parser.add_argument('--out')
    parser.add_argument('--exclude-ids')
    parser.add_argument('--batch-size', type=int, default=200_000)
    args = parser.parse_args()
    out = ROOT / (args.out or f'reports/audit_{args.mode}.json')
    if out.suffix != '.json':
        out = out / 'audit_summary.json'
    if args.mode == 'development':
        if not args.exclude_ids:
            parser.error('--mode development requires --exclude-ids with frozen holdout keys')
        result = audit_development(ROOT / args.data_dir, ROOT / args.exclude_ids, out,
                                   batch_size=args.batch_size)
        print(f"Audited {result['development_rows']:,} development rows; holdout excluded")
    else:
        result = audit_structure(ROOT / args.data_dir, out, batch_size=args.batch_size)
        print('Structure audit completed without target-label statistics')
    print(out)


if __name__ == '__main__':
    main()

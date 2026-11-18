#!/usr/bin/env python3
"""Freeze F4 time-split keys without training or reading target values."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f4.splits import make_splits


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/core")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="artifacts/splits")
    parser.add_argument("--report", help="Optional report path; defaults to reports/splits/<split_id>.json")
    args = parser.parse_args()
    result = make_splits(project_path(args.data_dir), project_path(args.config),
                         project_path(args.out), project_path(args.report) if args.report else None)
    print(json.dumps({"split_id": result["split_id"], "status": result["status"],
                      "holdout_n": result["holdout"]["n"], "folds": list(result["folds"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()

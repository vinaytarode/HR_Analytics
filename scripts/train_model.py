#!/usr/bin/env python3
"""
Train the attrition model and save .pkl artifacts.

Usage (from project root):
    python scripts/train_model.py
    python scripts/train_model.py --source csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.train_model import TrainingError, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train HR attrition model and save artifacts.")
    parser.add_argument(
        "--source",
        choices=["sqlite", "csv"],
        default="sqlite",
        help="Load training data from SQLite (default) or CSV file.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV path when --source csv.",
    )
    args = parser.parse_args()

    try:
        result = run_training(source=args.source, csv_path=args.csv)
        metrics = result["metrics"]
        print("Training complete.")
        print(f"  Model   : {result['model_path']}")
        print(f"  Scaler  : {result['scaler_path']}")
        print(f"  Features: {result['feature_columns_path']}")
        print(f"  Report  : {result['training_report_path']}")
        print(f"\n  Test accuracy: {metrics['test_accuracy'] * 100:.2f}%")
        print(f"  Train rows   : {metrics['train_rows']}")
        print(f"  Test rows    : {metrics['test_rows']}")
        print(f"  Feature count: {metrics['feature_count']}")
        return 0
    except TrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

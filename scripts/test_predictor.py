#!/usr/bin/env python3
"""
Test predictions without Streamlit (Phase 5).

Run:
    python scripts/test_predictor.py
    python scripts/test_predictor.py --employee-number 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.database import fetch_employee_by_number  # noqa: E402
from utils.predictor import (  # noqa: E402
    PredictionError,
    format_prediction_summary,
    predict_employee,
)
from utils.preprocess import INPUT_COLUMNS  # noqa: E402


def build_input_from_db(employee_number: int) -> dict:
    employee = fetch_employee_by_number(employee_number)
    if employee is None:
        raise PredictionError(f"Employee #{employee_number} not found in database.")
    return {col: employee[col] for col in INPUT_COLUMNS}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Test attrition predictor without Streamlit.")
    parser.add_argument(
        "--employee-number",
        type=int,
        default=1,
        help="EmployeeNumber from database to use as sample input.",
    )
    args = parser.parse_args()

    try:
        employee_input = build_input_from_db(args.employee_number)
        result = predict_employee(employee_input)

        print(f"\nEmployee #{args.employee_number}")
        print(f"  Risk message          : {result.risk_message}")
        print(f"  Prediction (encoded)  : {result.prediction_label}")
        print(f"  Prediction (text)     : {result.prediction_text}")
        print(f"  Confidence score      : {result.confidence_score:.2%}")
        print(f"  Attrition probability : {result.attrition_probability:.2%}")
        print(f"\n  Summary: {format_prediction_summary(result)}")
        print("\nPredictor test passed.")
        return 0

    except PredictionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Standalone tests for utils/preprocess.py (Phase 4).

Run:
    python scripts/test_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.database import fetch_employee_by_number  # noqa: E402
from utils.preprocess import (  # noqa: E402
    PreprocessingError,
    load_feature_columns,
    preprocess_for_prediction,
    preprocess_for_training,
)
from utils.train_model import load_raw_employee_data, load_scaler  # noqa: E402


def test_batch_training_shape() -> None:
    raw = load_raw_employee_data(source="sqlite")
    processed = preprocess_for_training(raw)
    assert processed.shape == (1470, 45), f"Unexpected shape: {processed.shape}"
    print("OK  batch training shape (1470, 45)")


def test_batch_vs_single_row() -> None:
    """One row from batch preprocessing must match single-row inference."""
    raw = load_raw_employee_data(source="sqlite")
    batch = preprocess_for_training(raw)

    employee = fetch_employee_by_number(1)
    if employee is None:
        raise PreprocessingError("Employee 1 not found in database.")

    # Build input dict without target / dropped columns
    from utils.preprocess import INPUT_COLUMNS

    employee_input = {col: employee[col] for col in INPUT_COLUMNS}
    single = preprocess_for_prediction(employee_input)

    feature_columns = load_feature_columns()
    batch_row = batch.drop(columns=["Attrition"]).iloc[0:1]
    batch_row = batch_row.reindex(columns=feature_columns, fill_value=0)

    if not np.allclose(batch_row.values, single.values):
        diff = (batch_row.values != single.values).sum()
        raise AssertionError(f"Batch vs single mismatch in {diff} cells.")

    print("OK  batch row matches single-row inference (employee #1)")


def test_scaled_prediction_shape() -> None:
    from utils.preprocess import preprocess_and_scale_for_prediction

    employee = fetch_employee_by_number(10)
    assert employee is not None
    from utils.preprocess import INPUT_COLUMNS

    payload = {col: employee[col] for col in INPUT_COLUMNS}
    scaler = load_scaler()
    scaled = preprocess_and_scale_for_prediction(payload, scaler)
    assert scaled.shape == (1, 44), f"Unexpected scaled shape: {scaled.shape}"
    print("OK  scaled inference shape (1, 44)")


def main() -> int:
    try:
        test_batch_training_shape()
        test_batch_vs_single_row()
        test_scaled_prediction_shape()
        print("\nAll preprocessing tests passed.")
        return 0
    except (PreprocessingError, AssertionError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

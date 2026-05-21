#!/usr/bin/env python3
"""
Test predict + SQLite storage integration (Phase 7).

Run:
    python scripts/test_prediction_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.database import fetch_prediction_history, get_employee_count  # noqa: E402
from utils.prediction_service import PredictionServiceError, predict_and_store  # noqa: E402
from utils.preprocess import INPUT_COLUMNS  # noqa: E402
from utils.database import fetch_employee_by_number  # noqa: E402


def main() -> int:
    if get_employee_count() == 0:
        print("ERROR: Database empty. Run: python scripts/init_database.py", file=sys.stderr)
        return 1

    employee = fetch_employee_by_number(5)
    if employee is None:
        print("ERROR: Employee #5 not found.", file=sys.stderr)
        return 1

    payload = {col: employee[col] for col in INPUT_COLUMNS}

    try:
        stored = predict_and_store(payload)
    except PredictionServiceError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    history = fetch_prediction_history(limit=5)
    latest = history.iloc[0]

    print("Prediction stored successfully.")
    print(f"  Record id        : {stored.record_id}")
    print(f"  Risk message     : {stored.result.risk_message}")
    print(f"  Confidence       : {stored.result.confidence_score:.2%}")
    print(f"  Latest DB label  : {latest['prediction_label']}")
    print(f"  History rows (5) : {len(history)}")

    assert int(latest["id"]) == stored.record_id
    print("\nPhase 7 integration test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

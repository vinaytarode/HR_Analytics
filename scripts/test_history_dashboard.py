#!/usr/bin/env python3
"""Test history dashboard data loaders (Phase 8)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.history_dashboard import (  # noqa: E402
    get_confidence_by_outcome,
    get_history_table,
    get_outcome_counts,
    get_summary_metrics,
)


def main() -> int:
    metrics = get_summary_metrics()
    table = get_history_table(limit=10)
    outcomes = get_outcome_counts()
    confidence = get_confidence_by_outcome()

    print("Summary:", metrics)
    print("Table rows:", len(table))
    print("Outcome counts:\n", outcomes)
    print("Confidence by outcome:\n", confidence)
    print("\nPhase 8 dashboard data test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

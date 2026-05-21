#!/usr/bin/env python3
"""
Bootstrap script: create SQLite tables and import CSV data.

Run from anywhere:
    python scripts/init_database.py

Or from the project root:
    python -m scripts.init_database
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `from utils.database import ...` when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.database import (  # noqa: E402
    DEFAULT_CSV_PATH,
    DB_PATH,
    DatabaseError,
    get_employee_count,
    import_employees_from_csv,
    init_database,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize SQLite database and import HR employee CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to CSV file (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--skip-import",
        action="store_true",
        help="Only create tables; do not import CSV rows.",
    )
    args = parser.parse_args()

    try:
        print(f"Project root : {PROJECT_ROOT}")
        print(f"Database path: {DB_PATH}")

        init_database()
        print("Tables created (or already exist).")

        if args.skip_import:
            print("CSV import skipped (--skip-import).")
            return 0

        row_count = import_employees_from_csv(args.csv)
        print(f"Imported {row_count} employee rows from {args.csv}")

        # Quick verification
        count = get_employee_count()
        print(f"employee_data row count: {count}")
        return 0

    except DatabaseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

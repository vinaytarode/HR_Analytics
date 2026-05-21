"""
SQLite database utilities for the HR Attrition Prediction application.

Responsibilities:
    - Open/close connections to database/employees.db
    - Create tables (employee_data, prediction_history)
    - Import CSV rows into employee_data
    - Insert and fetch prediction history records

Uses the built-in sqlite3 module (no ORM) for clarity and local deployment.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (always relative to project root, not the current working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "database" / "employees.db"
SCHEMA_PATH = PROJECT_ROOT / "database" / "schema.sql"
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "HR_Attrition.csv"

# Column order matches data/HR_Attrition.csv exactly
EMPLOYEE_COLUMNS: list[str] = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]


class DatabaseError(Exception):
    """Raised when a database operation fails."""


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string for timestamps."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_database_directory() -> None:
    """Create the database/ folder if it does not exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db_connection(*, write_transaction: bool = False) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that opens a SQLite connection and commits on success.

    Args:
        write_transaction: If True, starts with BEGIN IMMEDIATE for safer writes
            when multiple Streamlit sessions may write concurrently.

    Yields:
        sqlite3.Connection with row_factory=sqlite3.Row for dict-like access.

    Raises:
        DatabaseError: If connection or transaction fails.
    """
    ensure_database_directory()
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        if write_transaction:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError(f"Database operation failed: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def _read_schema_sql() -> str:
    """Load CREATE TABLE statements from database/schema.sql."""
    if not SCHEMA_PATH.exists():
        raise DatabaseError(f"Schema file not found: {SCHEMA_PATH}")
    return SCHEMA_PATH.read_text(encoding="utf-8")


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Execute schema.sql to create employee_data and prediction_history tables.

    Uses IF NOT EXISTS, so this function is safe to call multiple times.
    """
    schema_sql = _read_schema_sql()
    try:
        conn.executescript(schema_sql)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not create tables: {exc}") from exc


def init_database() -> None:
    """Create database file and all required tables."""
    with get_db_connection() as conn:
        create_tables(conn)


def get_employee_count(conn: sqlite3.Connection | None = None) -> int:
    """Return how many rows exist in employee_data."""
    query = "SELECT COUNT(*) AS count FROM employee_data"

    if conn is not None:
        row = conn.execute(query).fetchone()
        return int(row["count"])

    with get_db_connection() as connection:
        row = connection.execute(query).fetchone()
        return int(row["count"])


def import_employees_from_csv(
    csv_path: Path | str | None = None,
    *,
    replace_existing: bool = True,
) -> int:
    """
    Load HR_Attrition.csv into the employee_data table.

    Args:
        csv_path: Path to CSV file. Defaults to data/HR_Attrition.csv.
        replace_existing: If True, clears employee_data before import.

    Returns:
        Number of rows inserted.

    Raises:
        DatabaseError: If file is missing, columns differ, or insert fails.
    """
    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if not path.exists():
        raise DatabaseError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    missing = [col for col in EMPLOYEE_COLUMNS if col not in df.columns]
    if missing:
        raise DatabaseError(f"CSV is missing required columns: {missing}")

    extra = [col for col in df.columns if col not in EMPLOYEE_COLUMNS]
    if extra:
        raise DatabaseError(f"CSV has unexpected columns: {extra}")

    # Keep only known columns in a stable order
    df = df[EMPLOYEE_COLUMNS]

    placeholders = ", ".join(["?"] * len(EMPLOYEE_COLUMNS))
    column_list = ", ".join(EMPLOYEE_COLUMNS)
    insert_sql = f"""
        INSERT OR REPLACE INTO employee_data ({column_list})
        VALUES ({placeholders})
    """

    rows: Iterable[tuple[Any, ...]] = [
        tuple(record[col] for col in EMPLOYEE_COLUMNS)
        for record in df.to_dict(orient="records")
    ]

    with get_db_connection() as conn:
        create_tables(conn)
        if replace_existing:
            conn.execute("DELETE FROM employee_data")
        conn.executemany(insert_sql, list(rows))
        inserted = get_employee_count(conn)

    return inserted


def fetch_all_employees(limit: int | None = None) -> pd.DataFrame:
    """
    Fetch employee records from SQLite as a pandas DataFrame.

    Args:
        limit: Optional maximum number of rows (newest id first when limited).
    """
    base_query = f"""
        SELECT {", ".join(EMPLOYEE_COLUMNS)}
        FROM employee_data
        ORDER BY EmployeeNumber
    """
    if limit is not None:
        base_query += f" LIMIT {int(limit)}"

    with get_db_connection() as conn:
        df = pd.read_sql_query(base_query, conn)

    return df


def fetch_employee_by_number(employee_number: int) -> dict[str, Any] | None:
    """Return one employee row as a dictionary, or None if not found."""
    query = f"""
        SELECT {", ".join(EMPLOYEE_COLUMNS)}
        FROM employee_data
        WHERE EmployeeNumber = ?
    """
    with get_db_connection() as conn:
        row = conn.execute(query, (employee_number,)).fetchone()

    if row is None:
        return None
    return dict(row)


def _validate_prediction_payload(
    *,
    prediction_label: str,
    confidence_score: float,
    risk_message: str,
    employee_input: dict[str, Any],
) -> None:
    """Validate values before writing to prediction_history."""
    if not prediction_label:
        raise DatabaseError("prediction_label cannot be empty.")
    if not risk_message:
        raise DatabaseError("risk_message cannot be empty.")
    if not employee_input:
        raise DatabaseError("employee_input cannot be empty.")
    if not 0.0 <= confidence_score <= 1.0:
        raise DatabaseError("confidence_score must be between 0.0 and 1.0.")


def insert_prediction(
    *,
    prediction_label: str,
    confidence_score: float,
    risk_message: str,
    employee_input: dict[str, Any],
    created_at: str | None = None,
) -> int:
    """
    Store one prediction in prediction_history inside a write transaction.

    Args:
        prediction_label: Model class label (e.g. 'Yes' / 'No').
        confidence_score: Probability of predicted class (0.0–1.0).
        risk_message: Human-readable message for the UI.
        employee_input: Dictionary of form values (stored as JSON).
        created_at: Optional ISO timestamp; defaults to current UTC time.

    Returns:
        The new row's id.
    """
    _validate_prediction_payload(
        prediction_label=prediction_label,
        confidence_score=confidence_score,
        risk_message=risk_message,
        employee_input=employee_input,
    )

    timestamp = created_at or _utc_now_iso()
    payload = json.dumps(employee_input, sort_keys=True)

    query = """
        INSERT INTO prediction_history (
            created_at,
            prediction_label,
            confidence_score,
            risk_message,
            employee_input
        )
        VALUES (?, ?, ?, ?, ?)
    """

    with get_db_connection(write_transaction=True) as conn:
        create_tables(conn)
        cursor = conn.execute(
            query,
            (timestamp, prediction_label, confidence_score, risk_message, payload),
        )
        record_id = int(cursor.lastrowid)
        if record_id <= 0:
            raise DatabaseError("Insert succeeded but no prediction id was returned.")
        return record_id


def fetch_prediction_history(limit: int = 100) -> pd.DataFrame:
    """
    Fetch recent predictions for the Streamlit history dashboard.

    Returns:
        DataFrame with parsed employee_input JSON expanded for display.
    """
    query = """
        SELECT
            id,
            created_at,
            prediction_label,
            confidence_score,
            risk_message,
            employee_input
        FROM prediction_history
        ORDER BY datetime(created_at) DESC
        LIMIT ?
    """

    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=(int(limit),))

    if df.empty:
        return df

    # Parse JSON so the UI can show individual fields later (Phase 8)
    input_df = pd.json_normalize(df["employee_input"].apply(json.loads))
    df = df.drop(columns=["employee_input"])
    return pd.concat([df, input_df], axis=1)


def fetch_prediction_summary() -> dict[str, Any]:
    """Return simple counts for dashboard statistics (Phase 8)."""
    query = """
        SELECT
            COUNT(*) AS total_predictions,
            AVG(confidence_score) AS avg_confidence,
            SUM(CASE WHEN prediction_label IN ('Yes', '1', 1) THEN 1 ELSE 0 END)
                AS attrition_predictions
        FROM prediction_history
    """
    with get_db_connection() as conn:
        row = conn.execute(query).fetchone()

    return {
        "total_predictions": int(row["total_predictions"] or 0),
        "avg_confidence": float(row["avg_confidence"] or 0.0),
        "attrition_predictions": int(row["attrition_predictions"] or 0),
    }

"""
Data preparation helpers for the prediction history dashboard (Phase 8).

Reads from SQLite via utils/database.py and formats results for Streamlit widgets.
"""

from __future__ import annotations

import pandas as pd

from utils.database import DatabaseError, fetch_prediction_history, fetch_prediction_summary

# Columns shown in the main history table (keeps the UI readable)
DISPLAY_COLUMNS: list[str] = [
    "id",
    "created_at",
    "prediction_label",
    "confidence_score",
    "risk_message",
    "Age",
    "Department",
    "JobRole",
    "OverTime",
    "MonthlyIncome",
]


def get_summary_metrics() -> dict[str, float | int]:
    """
    Fetch aggregate statistics from prediction_history.

    Returns:
        total_predictions, attrition_predictions, stay_predictions,
        attrition_rate_pct, avg_confidence
    """
    summary = fetch_prediction_summary()
    total = int(summary["total_predictions"])
    attrition = int(summary["attrition_predictions"])
    stay = max(total - attrition, 0)
    rate = (attrition / total * 100.0) if total > 0 else 0.0

    return {
        "total_predictions": total,
        "attrition_predictions": attrition,
        "stay_predictions": stay,
        "attrition_rate_pct": round(rate, 1),
        "avg_confidence": round(float(summary["avg_confidence"]) * 100, 1),
    }


def get_history_table(limit: int = 50) -> pd.DataFrame:
    """
    Load recent predictions from SQLite and format for st.dataframe().

    Data flow:
        prediction_history table
            → fetch_prediction_history()  (SQL + JSON parse)
            → select/rename columns
            → formatted display DataFrame
    """
    history = fetch_prediction_history(limit=limit)
    if history.empty:
        return history

    available = [col for col in DISPLAY_COLUMNS if col in history.columns]
    table = history[available].copy()

    # Friendly column names for non-technical users
    table.rename(
        columns={
            "id": "Record ID",
            "created_at": "Timestamp (UTC)",
            "prediction_label": "Attrition",
            "confidence_score": "Confidence",
            "risk_message": "Risk message",
            "Age": "Age",
            "Department": "Department",
            "JobRole": "Job role",
            "OverTime": "Over time",
            "MonthlyIncome": "Monthly income",
        },
        inplace=True,
    )

    if "Confidence" in table.columns:
        table["Confidence"] = table["Confidence"].apply(lambda value: f"{float(value) * 100:.1f}%")

    return table


def get_outcome_counts() -> pd.DataFrame:
    """Return counts of Yes/No predictions for bar charts."""
    history = fetch_prediction_history(limit=500)
    if history.empty:
        return pd.DataFrame(columns=["Outcome", "Count"])

    counts = history["prediction_label"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    return counts


def get_confidence_by_outcome() -> pd.DataFrame:
    """Average confidence grouped by predicted attrition label."""
    history = fetch_prediction_history(limit=500)
    if history.empty:
        return pd.DataFrame(columns=["prediction_label", "confidence_score"])

    grouped = (
        history.groupby("prediction_label", as_index=False)["confidence_score"]
        .mean()
        .rename(columns={"confidence_score": "avg_confidence"})
    )
    grouped["avg_confidence"] = grouped["avg_confidence"].round(3)
    return grouped

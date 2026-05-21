"""
End-to-end prediction workflow: ML inference + SQLite persistence.

Used by app.py when the user clicks "Predict Attrition".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from utils.database import DatabaseError, insert_prediction
from utils.predictor import PredictionError, PredictionResult, predict_employee

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredPrediction:
    """Prediction result plus database record metadata."""

    result: PredictionResult
    record_id: int
    created_at: str | None = None


class PredictionServiceError(Exception):
    """Raised when predict-and-store workflow fails."""


def predict_and_store(employee_input: dict[str, Any]) -> StoredPrediction:
    """
    Run model inference and save the outcome to prediction_history.

    Flow:
        1. predict_employee()  — ML pipeline
        2. insert_prediction() — SQLite write (transaction-safe)

    Args:
        employee_input: Raw form dictionary (30 INPUT_COLUMNS fields).

    Returns:
        StoredPrediction with model output and new database row id.

    Raises:
        PredictionServiceError: On prediction or database failure.
    """
    try:
        logger.info("Starting prediction for employee input.")
        result = predict_employee(employee_input)
    except PredictionError as exc:
        logger.error("Prediction failed: %s", exc)
        raise PredictionServiceError(str(exc)) from exc

    try:
        logger.info("Saving prediction to SQLite.")
        record_id = insert_prediction(
            prediction_label=result.prediction_text,
            confidence_score=result.confidence_score,
            risk_message=result.risk_message,
            employee_input=employee_input,
        )
    except DatabaseError as exc:
        logger.error("Failed to store prediction: %s", exc)
        raise PredictionServiceError(
            f"Prediction completed ({result.risk_message}) but could not be saved: {exc}"
        ) from exc

    logger.info("Prediction stored with id=%s", record_id)
    return StoredPrediction(result=result, record_id=record_id)

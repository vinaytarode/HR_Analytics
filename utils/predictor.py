"""
Prediction service for HR employee attrition.

Loads trained artifacts, preprocesses user input, and returns:
    - numeric prediction (0 = stay, 1 = leave)
    - human-readable risk message
    - confidence score (probability of the predicted class)
    - attrition probability P(leave)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.preprocess import PreprocessingError, preprocess_and_scale_for_prediction
from utils.train_model import TrainingError, load_model, load_scaler

logger = logging.getLogger(__name__)

# Human-readable labels aligned with training target encoding
CLASS_LABELS = {0: "No", 1: "Yes"}
RISK_MESSAGES = {
    0: "Low Risk of Attrition",
    1: "High Risk of Attrition",
}


class PredictionError(Exception):
    """Raised when prediction cannot be completed."""


@dataclass(frozen=True)
class PredictionResult:
    """Structured prediction output for Streamlit and database storage."""

    prediction_label: int
    prediction_text: str
    risk_message: str
    confidence_score: float
    attrition_probability: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary (easy to store as JSON in SQLite)."""
        return asdict(self)


class AttritionPredictor:
    """
    Loads model artifacts once and serves repeated predictions.

    Usage:
        predictor = AttritionPredictor()
        predictor.load_artifacts()
        result = predictor.predict(employee_input)
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._scaler is not None

    def load_artifacts(self) -> None:
        """Load model and scaler from models/ directory."""
        try:
            logger.info("Loading attrition model and scaler.")
            self._model = load_model()
            self._scaler = load_scaler()
            logger.info("Model artifacts loaded successfully.")
        except TrainingError as exc:
            raise PredictionError(str(exc)) from exc

    def predict(self, employee_input: dict[str, Any]) -> PredictionResult:
        """
        Run end-to-end prediction for one employee record.

        Args:
            employee_input: Dictionary with keys from preprocess.INPUT_COLUMNS.

        Returns:
            PredictionResult with label, messages, and probabilities.
        """
        if not self.is_loaded:
            self.load_artifacts()

        assert self._model is not None
        assert self._scaler is not None

        try:
            logger.debug("Preprocessing employee input for prediction.")
            scaled_features = preprocess_and_scale_for_prediction(
                employee_input,
                self._scaler,
            )

            prediction = int(self._model.predict(scaled_features)[0])
            probabilities = self._model.predict_proba(scaled_features)[0]

            # Class order for binary LogisticRegression: [0, 1]
            attrition_probability = float(probabilities[1])
            confidence_score = float(probabilities[prediction])

            result = PredictionResult(
                prediction_label=prediction,
                prediction_text=CLASS_LABELS[prediction],
                risk_message=RISK_MESSAGES[prediction],
                confidence_score=round(confidence_score, 4),
                attrition_probability=round(attrition_probability, 4),
            )
            logger.info(
                "Prediction complete: label=%s confidence=%.4f attrition_prob=%.4f",
                result.prediction_text,
                result.confidence_score,
                result.attrition_probability,
            )
            return result

        except PreprocessingError as exc:
            logger.error("Preprocessing failed: %s", exc)
            raise PredictionError(f"Invalid employee input: {exc}") from exc
        except Exception as exc:
            logger.exception("Unexpected error during prediction.")
            raise PredictionError(f"Prediction failed: {exc}") from exc


# Module-level singleton for simple imports from Streamlit
_default_predictor = AttritionPredictor()


def predict_employee(employee_input: dict[str, Any]) -> PredictionResult:
    """
    Convenience function used by app.py (Phase 6–7).

    Example:
        result = predict_employee({"Age": 35, "Gender": "Male", ...})
        print(result.risk_message, result.confidence_score)
    """
    return _default_predictor.predict(employee_input)


def format_prediction_summary(result: PredictionResult) -> str:
    """Single-line summary string for logs or simple UI display."""
    return (
        f"{result.risk_message} "
        f"(Attrition={result.prediction_text}, "
        f"confidence={result.confidence_score * 100:.1f}%)"
    )

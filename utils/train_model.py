"""
Train the attrition classifier and save production artifacts.

Uses utils/preprocess.py for all feature engineering (shared with inference).

Artifacts written to models/:
    - attrition_model.pkl
    - scaler.pkl
    - feature_columns.json
    - encoding_metadata.json
    - training_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.database import DEFAULT_CSV_PATH, fetch_all_employees
from utils.preprocess import (
    TARGET_COLUMN,
    preprocess_for_training,
    save_encoding_metadata,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "attrition_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42


class TrainingError(Exception):
    """Raised when training or artifact saving fails."""


def load_raw_employee_data(
    source: Literal["sqlite", "csv"] = "sqlite",
    csv_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Load the raw employee dataset.

    Args:
        source: 'sqlite' reads from employee_data; 'csv' reads HR_Attrition.csv.
        csv_path: Optional CSV path when source='csv'.
    """
    if source == "sqlite":
        df = fetch_all_employees()
        if df.empty:
            raise TrainingError(
                "employee_data table is empty. Run: python scripts/init_database.py"
            )
        return df

    path = Path(csv_path) if csv_path is not None else DEFAULT_CSV_PATH
    if not path.exists():
        raise TrainingError(f"CSV not found: {path}")
    return pd.read_csv(path)


def split_features_target(processed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix X and target vector y."""
    if TARGET_COLUMN not in processed_df.columns:
        raise TrainingError(f"Processed data must include '{TARGET_COLUMN}'.")
    X = processed_df.drop(columns=[TARGET_COLUMN])
    y = processed_df[TARGET_COLUMN]
    return X, y


def train_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[LogisticRegression, StandardScaler, dict[str, Any]]:
    """
    Train Logistic Regression with StandardScaler (notebook's best model).

    Returns:
        fitted model, fitted scaler, evaluation metrics dict
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)

    test_predictions = model.predict(X_test_scaled)
    metrics = {
        "model_name": "Logistic Regression",
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_count": int(X.shape[1]),
        "classification_report": classification_report(
            y_test, test_predictions, output_dict=True
        ),
    }
    return model, scaler, metrics


def save_artifacts(
    model: LogisticRegression,
    scaler: StandardScaler,
    feature_columns: list[str],
    metrics: dict[str, Any],
) -> None:
    """Persist model, scaler, and feature column order to models/."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(feature_columns, indent=2),
        encoding="utf-8",
    )

    report = {
        **metrics,
        "feature_columns_file": str(FEATURE_COLUMNS_PATH.name),
        "model_file": MODEL_PATH.name,
        "scaler_file": SCALER_PATH.name,
        "encoding_metadata_file": "encoding_metadata.json",
    }
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_model() -> LogisticRegression:
    """Load the trained classifier from disk."""
    if not MODEL_PATH.exists():
        raise TrainingError(
            f"Model file not found: {MODEL_PATH}. Run: python scripts/train_model.py"
        )
    return joblib.load(MODEL_PATH)


def load_scaler() -> StandardScaler:
    """Load the fitted StandardScaler from disk."""
    if not SCALER_PATH.exists():
        raise TrainingError(
            f"Scaler file not found: {SCALER_PATH}. Run: python scripts/train_model.py"
        )
    return joblib.load(SCALER_PATH)


def load_feature_columns() -> list[str]:
    """Load the exact feature column order used during training."""
    from utils.preprocess import load_feature_columns as _load_feature_columns

    try:
        return _load_feature_columns()
    except Exception as exc:
        raise TrainingError(str(exc)) from exc


def run_training(
    source: Literal["sqlite", "csv"] = "sqlite",
    csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """
    End-to-end training pipeline: load → preprocess → train → save.

    Returns:
        Dictionary with paths and evaluation metrics.
    """
    raw_df = load_raw_employee_data(source=source, csv_path=csv_path)
    processed_df = preprocess_for_training(raw_df)
    X, y = split_features_target(processed_df)

    expected_shape = (len(raw_df), 45)
    if processed_df.shape != expected_shape:
        raise TrainingError(
            f"Unexpected processed shape {processed_df.shape}; "
            f"expected {expected_shape}. Check preprocessing consistency."
        )

    save_encoding_metadata(raw_df)

    model, scaler, metrics = train_logistic_regression(X, y)
    feature_columns = X.columns.tolist()
    save_artifacts(model, scaler, feature_columns, metrics)

    return {
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "feature_columns_path": str(FEATURE_COLUMNS_PATH),
        "training_report_path": str(TRAINING_REPORT_PATH),
        "metrics": metrics,
    }

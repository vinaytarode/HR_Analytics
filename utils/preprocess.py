"""
Preprocessing pipeline shared by training and inference.

CRITICAL RULE:
    Training and prediction must use the same functions in this module.
    Do not duplicate encoding logic elsewhere.

Notebook reference: hr_attrition_analysis.ipynb Sections 4.1–4.4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
ENCODING_METADATA_PATH = MODELS_DIR / "encoding_metadata.json"

# ---------------------------------------------------------------------------
# Constants (must match notebook + utils/train_model.py)
# ---------------------------------------------------------------------------
COLS_TO_DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
TARGET_COLUMN = "Attrition"

# Columns the Streamlit form will collect (raw values, before encoding)
INPUT_COLUMNS: list[str] = [
    "Age",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
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
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Multi-class categoricals that become one-hot columns
ONEHOT_SOURCE_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
]

# Binary categoricals label-encoded by the notebook
BINARY_COLUMNS = ["Gender", "OverTime"]

# Allowed string options (helps Streamlit dropdowns and validation)
CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "BusinessTravel": ["Non-Travel", "Travel_Frequently", "Travel_Rarely"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": [
        "Human Resources",
        "Life Sciences",
        "Marketing",
        "Medical",
        "Other",
        "Technical Degree",
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative",
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}


class PreprocessingError(Exception):
    """Raised when input data cannot be transformed safely."""


def load_feature_columns() -> list[str]:
    """Load trained feature names in the exact order expected by the model."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise PreprocessingError(
            f"Missing {FEATURE_COLUMNS_PATH}. Run: python scripts/train_model.py"
        )
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def load_encoding_metadata() -> dict[str, Any]:
    """Load binary encoding maps saved during training."""
    if not ENCODING_METADATA_PATH.exists():
        raise PreprocessingError(
            f"Missing {ENCODING_METADATA_PATH}. Run: python scripts/train_model.py"
        )
    return json.loads(ENCODING_METADATA_PATH.read_text(encoding="utf-8"))


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove ID / constant columns (notebook Section 4.1)."""
    data = df.copy()
    existing_drops = [col for col in COLS_TO_DROP if col in data.columns]
    data.drop(columns=existing_drops, inplace=True)
    return data


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill nulls using mode (text) or median (numeric) — notebook Section 4.2."""
    data = df.copy()
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)
    return data


def encode_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Attrition Yes/No to 1/0 (notebook Section 4.3)."""
    data = df.copy()
    if TARGET_COLUMN not in data.columns:
        raise PreprocessingError(f"Missing target column: {TARGET_COLUMN}")

    data[TARGET_COLUMN] = data[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if data[TARGET_COLUMN].isnull().any():
        raise PreprocessingError(
            f"'{TARGET_COLUMN}' must contain only 'Yes' or 'No' before encoding."
        )
    return data


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categoricals exactly like the notebook (Section 4.4).

    - Binary columns → LabelEncoder (one shared instance, refit per column)
    - Multi-class columns → pandas get_dummies(drop_first=True)
    """
    data = df.copy()
    label_encoder = LabelEncoder()
    categorical_columns = data.select_dtypes(include=["object", "string"]).columns.tolist()

    for col in categorical_columns:
        if col == TARGET_COLUMN:
            continue
        if data[col].nunique() == 2:
            data[col] = label_encoder.fit_transform(data[col])

    # One-hot encode remaining multi-class columns after binary encoding
    data = apply_one_hot_encoding_batch(data)

    remaining_objects = data.select_dtypes(include=["object", "string"]).columns.tolist()
    if remaining_objects:
        raise PreprocessingError(f"Unencoded categorical columns remain: {remaining_objects}")

    return data


def build_binary_encoding_maps(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Capture label mappings for binary columns using the same logic as training.

    Saved to models/encoding_metadata.json so inference uses identical codes.
    """
    maps: dict[str, dict[str, int]] = {}
    cleaned = drop_unused_columns(df)
    cleaned = fill_missing_values(cleaned)

    label_encoder = LabelEncoder()
    for col in BINARY_COLUMNS:
        if col not in cleaned.columns:
            continue
        label_encoder.fit(cleaned[col].astype(str))
        maps[col] = {
            str(category): int(code)
            for category, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        }
    return maps


def save_encoding_metadata(df: pd.DataFrame) -> None:
    """Persist binary encoding maps created from the training dataset."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "binary_maps": build_binary_encoding_maps(df),
        "onehot_source_columns": ONEHOT_SOURCE_COLUMNS,
        "input_columns": INPUT_COLUMNS,
    }
    ENCODING_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def apply_binary_encoding(df: pd.DataFrame, binary_maps: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Apply saved binary maps to a DataFrame (used at inference)."""
    data = df.copy()
    for col, mapping in binary_maps.items():
        if col not in data.columns:
            raise PreprocessingError(f"Missing binary column: {col}")
        values = data[col].astype(str)
        unknown = sorted(set(values.unique()) - set(mapping.keys()))
        if unknown:
            raise PreprocessingError(
                f"Unknown values in '{col}': {unknown}. Allowed: {sorted(mapping.keys())}"
            )
        data[col] = values.map(mapping)
    return data


def apply_one_hot_encoding_batch(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode multi-class columns on a full dataset (training)."""
    data = df.copy()
    for col in ONEHOT_SOURCE_COLUMNS:
        if col not in data.columns:
            raise PreprocessingError(f"Missing one-hot source column: {col}")
        dummies = pd.get_dummies(data[col].astype(str), prefix=col, drop_first=True)
        data = pd.concat([data, dummies], axis=1)
        data.drop(columns=[col], inplace=True)
    return data


def apply_one_hot_encoding_inference(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    One-hot encode a single employee row using training column names.

    Why not get_dummies on one row?
        With drop_first=True, a single category row produces no dummy columns.
        We set dummy columns explicitly from models/feature_columns.json.
    """
    data = df.copy()
    for col in ONEHOT_SOURCE_COLUMNS:
        if col not in data.columns:
            raise PreprocessingError(f"Missing one-hot source column: {col}")

        value = str(data[col].iloc[0])
        dummy_columns = [name for name in feature_columns if name.startswith(f"{col}_")]

        for dummy_col in dummy_columns:
            data[dummy_col] = 0

        active_dummy = f"{col}_{value}"
        if active_dummy in dummy_columns:
            data[active_dummy] = 1
        elif dummy_columns:
            # Value is the reference (dropped) category → all zeros (correct for drop_first)
            pass
        else:
            raise PreprocessingError(
                f"Unexpected category '{value}' for column '{col}'. "
                f"Expected one of: {CATEGORICAL_OPTIONS.get(col, 'see metadata')}"
            )

        data.drop(columns=[col], inplace=True)

    return data


def align_feature_columns(
    features: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Reindex columns to match training order.

    Missing one-hot columns (reference category dropped during training)
    are filled with 0.
    """
    aligned = features.reindex(columns=feature_columns, fill_value=0)
    return aligned


def preprocess_for_training(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full batch preprocessing used during model training.

    Returns numeric DataFrame including encoded Attrition target.
    """
    data = drop_unused_columns(df)
    data = fill_missing_values(data)
    data = encode_target_column(data)
    data = encode_categorical_columns(data)
    return data


def validate_employee_input(employee: dict[str, Any]) -> None:
    """Ensure required fields exist before inference preprocessing."""
    missing = [col for col in INPUT_COLUMNS if col not in employee]
    if missing:
        raise PreprocessingError(f"Missing required input fields: {missing}")


def employee_dict_to_dataframe(employee: dict[str, Any]) -> pd.DataFrame:
    """Convert a single employee dictionary into a one-row DataFrame."""
    validate_employee_input(employee)
    row = {col: employee[col] for col in INPUT_COLUMNS}
    return pd.DataFrame([row])


def preprocess_for_prediction(employee: dict[str, Any]) -> pd.DataFrame:
    """
    Transform one employee record into the 44 model features (unscaled).

    Steps:
        1. Validate input fields
        2. Drop unused columns (none in form, but keeps parity)
        3. Fill missing values
        4. Binary + one-hot encoding using training metadata
        5. Align columns to models/feature_columns.json
    """
    feature_columns = load_feature_columns()
    metadata = load_encoding_metadata()
    binary_maps = metadata["binary_maps"]

    frame = employee_dict_to_dataframe(employee)
    frame = fill_missing_values(frame)
    frame = apply_binary_encoding(frame, binary_maps)
    frame = apply_one_hot_encoding_inference(frame, feature_columns)

    aligned = align_feature_columns(frame, feature_columns)
    return aligned


def scale_features(
    features: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    Apply the fitted StandardScaler (transform only — never fit at inference).

    Returns a NumPy array shaped (n_samples, n_features).
    """
    return scaler.transform(features)


def preprocess_and_scale_for_prediction(
    employee: dict[str, Any],
    scaler: StandardScaler,
) -> np.ndarray:
    """Convenience helper: raw dict → scaled NumPy array ready for model.predict."""
    features = preprocess_for_prediction(employee)
    return scale_features(features, scaler)

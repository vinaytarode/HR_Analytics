"""
HR Attrition Prediction — Streamlit web application.

Phase 6: Employee input form + live prediction display.
Phase 7: Each prediction is saved to SQLite prediction_history.
Phase 8: Prediction history dashboard (SQLite-backed).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure project root is on sys.path when Streamlit runs app.py directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.database import DatabaseError  # noqa: E402
from utils.history_dashboard import (  # noqa: E402
    get_confidence_by_outcome,
    get_history_table,
    get_outcome_counts,
    get_summary_metrics,
)
from utils.prediction_service import (  # noqa: E402
    PredictionServiceError,
    StoredPrediction,
    predict_and_store,
)
from utils.predictor import PredictionResult, format_prediction_summary  # noqa: E402
from utils.preprocess import CATEGORICAL_OPTIONS, INPUT_COLUMNS  # noqa: E402
from utils.train_model import MODEL_PATH, SCALER_PATH  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light styling for a cleaner dashboard look
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem; }
        .risk-high { color: #d62728; font-weight: 700; }
        .risk-low { color: #2ca02c; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Default values (medians from HR_Attrition.csv — sensible starting point)
# ---------------------------------------------------------------------------
DEFAULT_VALUES: dict[str, Any] = {
    "Age": 36,
    "DailyRate": 802,
    "DistanceFromHome": 7,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "HourlyRate": 66,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyIncome": 4919,
    "MonthlyRate": 14235,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "JobRole": "Research Scientist",
    "MaritalStatus": "Married",
    "OverTime": "No",
}


def artifacts_ready() -> tuple[bool, list[str]]:
    """Check that trained model files exist before predicting."""
    missing = []
    if not MODEL_PATH.exists():
        missing.append(str(MODEL_PATH.relative_to(PROJECT_ROOT)))
    if not SCALER_PATH.exists():
        missing.append(str(SCALER_PATH.relative_to(PROJECT_ROOT)))
    return len(missing) == 0, missing


def load_employee_from_database(employee_number: int) -> dict[str, Any] | None:
    """Fetch one employee from SQLite to pre-fill the form."""
    try:
        from utils.database import fetch_employee_by_number

        record = fetch_employee_by_number(employee_number)
        if record is None:
            return None
        return {column: record[column] for column in INPUT_COLUMNS}
    except Exception as exc:
        st.sidebar.error(f"Database error: {exc}")
        return None


def render_sidebar() -> None:
    """Sidebar: setup status, instructions, optional DB sample loader."""
    st.sidebar.title("Navigation")
    st.sidebar.markdown(
        """
        **How to use**
        1. Enter employee details in the form  
        2. Click **Predict Attrition**  
        3. Review risk level and confidence score  

        Each prediction is saved to SQLite automatically.  
        Open the **Prediction History** tab to review past runs.
        """
    )

    ready, missing = artifacts_ready()
    if ready:
        st.sidebar.success("Model artifacts found")
    else:
        st.sidebar.error("Model not trained yet")
        st.sidebar.code("python scripts/train_model.py", language="bash")
        for path in missing:
            st.sidebar.caption(f"Missing: {path}")

    st.sidebar.divider()
    st.sidebar.subheader("Load sample employee")
    sample_id = st.sidebar.number_input(
        "Employee number",
        min_value=1,
        max_value=9999,
        value=1,
        step=1,
        help="Loads values from SQLite employee_data (Phase 2).",
    )
    if st.sidebar.button("Load into form", use_container_width=True):
        payload = load_employee_from_database(int(sample_id))
        if payload is None:
            st.sidebar.warning(f"Employee #{sample_id} not found.")
        else:
            st.session_state["form_defaults"] = payload
            st.sidebar.success(f"Loaded employee #{sample_id}")
            st.rerun()

    if st.sidebar.button("Reset to defaults", use_container_width=True):
        st.session_state["form_defaults"] = DEFAULT_VALUES.copy()
        st.session_state.pop("last_stored_prediction", None)
        st.rerun()


def _default(column: str) -> Any:
    """Read default widget value from session state or built-in defaults."""
    form_defaults = st.session_state.get("form_defaults", DEFAULT_VALUES)
    return form_defaults.get(column, DEFAULT_VALUES.get(column))


def _select_index(options: list[Any], value: Any) -> int:
    """Safe index lookup for st.selectbox defaults."""
    try:
        return options.index(value)
    except ValueError:
        return 0


def render_employee_form() -> dict[str, Any] | None:
    """
    Render the employee input form.

    Returns:
        Employee dictionary when submitted, otherwise None.
    """
    st.subheader("Employee details")
    st.caption("All fields match the training dataset used by the attrition model.")

    with st.form("employee_attrition_form", clear_on_submit=False):
        tab_personal, tab_job, tab_pay, tab_satisfaction = st.tabs(
            ["Personal", "Job", "Compensation", "Satisfaction & Tenure"]
        )

        # --- Personal tab ---
        with tab_personal:
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=18, max_value=65, value=int(_default("Age")))
                gender = st.selectbox(
                    "Gender",
                    CATEGORICAL_OPTIONS["Gender"],
                    index=_select_index(CATEGORICAL_OPTIONS["Gender"], _default("Gender")),
                )
                marital = st.selectbox(
                    "Marital status",
                    CATEGORICAL_OPTIONS["MaritalStatus"],
                    index=_select_index(CATEGORICAL_OPTIONS["MaritalStatus"], _default("MaritalStatus")),
                )
            with col2:
                education = st.slider("Education level", 1, 5, int(_default("Education")))
                education_field = st.selectbox(
                    "Education field",
                    CATEGORICAL_OPTIONS["EducationField"],
                    index=_select_index(CATEGORICAL_OPTIONS["EducationField"], _default("EducationField")),
                )
                distance_home = st.number_input(
                    "Distance from home",
                    min_value=1,
                    max_value=30,
                    value=int(_default("DistanceFromHome")),
                )
            with col3:
                business_travel = st.selectbox(
                    "Business travel",
                    CATEGORICAL_OPTIONS["BusinessTravel"],
                    index=_select_index(CATEGORICAL_OPTIONS["BusinessTravel"], _default("BusinessTravel")),
                )
                overtime = st.selectbox(
                    "Over time",
                    CATEGORICAL_OPTIONS["OverTime"],
                    index=_select_index(CATEGORICAL_OPTIONS["OverTime"], _default("OverTime")),
                )

        # --- Job tab ---
        with tab_job:
            col1, col2, col3 = st.columns(3)
            with col1:
                department = st.selectbox(
                    "Department",
                    CATEGORICAL_OPTIONS["Department"],
                    index=_select_index(CATEGORICAL_OPTIONS["Department"], _default("Department")),
                )
                job_role = st.selectbox(
                    "Job role",
                    CATEGORICAL_OPTIONS["JobRole"],
                    index=_select_index(CATEGORICAL_OPTIONS["JobRole"], _default("JobRole")),
                )
            with col2:
                job_level = st.slider("Job level", 1, 5, int(_default("JobLevel")))
                job_involvement = st.slider("Job involvement", 1, 4, int(_default("JobInvolvement")))
            with col3:
                job_satisfaction = st.slider("Job satisfaction", 1, 4, int(_default("JobSatisfaction")))
                performance = st.selectbox(
                    "Performance rating",
                    [3, 4],
                    index=_select_index([3, 4], int(_default("PerformanceRating"))),
                )

        # --- Compensation tab ---
        with tab_pay:
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_income = st.number_input(
                    "Monthly income",
                    min_value=1000,
                    max_value=20000,
                    value=int(_default("MonthlyIncome")),
                    step=100,
                )
                monthly_rate = st.number_input(
                    "Monthly rate",
                    min_value=2000,
                    max_value=27000,
                    value=int(_default("MonthlyRate")),
                    step=100,
                )
            with col2:
                daily_rate = st.number_input(
                    "Daily rate",
                    min_value=100,
                    max_value=1500,
                    value=int(_default("DailyRate")),
                )
                hourly_rate = st.number_input(
                    "Hourly rate",
                    min_value=30,
                    max_value=100,
                    value=int(_default("HourlyRate")),
                )
            with col3:
                percent_hike = st.slider(
                    "Percent salary hike",
                    11,
                    25,
                    int(_default("PercentSalaryHike")),
                )
                stock_options = st.slider("Stock option level", 0, 3, int(_default("StockOptionLevel")))

        # --- Satisfaction & tenure tab ---
        with tab_satisfaction:
            col1, col2, col3 = st.columns(3)
            with col1:
                env_sat = st.slider(
                    "Environment satisfaction",
                    1,
                    4,
                    int(_default("EnvironmentSatisfaction")),
                )
                rel_sat = st.slider(
                    "Relationship satisfaction",
                    1,
                    4,
                    int(_default("RelationshipSatisfaction")),
                )
                wlb = st.slider("Work-life balance", 1, 4, int(_default("WorkLifeBalance")))
            with col2:
                total_years = st.number_input(
                    "Total working years",
                    min_value=0,
                    max_value=40,
                    value=int(_default("TotalWorkingYears")),
                )
                years_company = st.number_input(
                    "Years at company",
                    min_value=0,
                    max_value=40,
                    value=int(_default("YearsAtCompany")),
                )
                years_role = st.number_input(
                    "Years in current role",
                    min_value=0,
                    max_value=20,
                    value=int(_default("YearsInCurrentRole")),
                )
            with col3:
                years_promo = st.number_input(
                    "Years since last promotion",
                    min_value=0,
                    max_value=15,
                    value=int(_default("YearsSinceLastPromotion")),
                )
                years_manager = st.number_input(
                    "Years with current manager",
                    min_value=0,
                    max_value=20,
                    value=int(_default("YearsWithCurrManager")),
                )
                num_companies = st.number_input(
                    "Number of companies worked",
                    min_value=0,
                    max_value=10,
                    value=int(_default("NumCompaniesWorked")),
                )
                training = st.slider(
                    "Training times last year",
                    0,
                    6,
                    int(_default("TrainingTimesLastYear")),
                )

        submitted = st.form_submit_button("Predict Attrition", type="primary", use_container_width=True)

    if not submitted:
        return None

    return {
        "Age": age,
        "BusinessTravel": business_travel,
        "DailyRate": daily_rate,
        "Department": department,
        "DistanceFromHome": distance_home,
        "Education": education,
        "EducationField": education_field,
        "EnvironmentSatisfaction": env_sat,
        "Gender": gender,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role,
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime,
        "PercentSalaryHike": percent_hike,
        "PerformanceRating": performance,
        "RelationshipSatisfaction": rel_sat,
        "StockOptionLevel": stock_options,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager": years_manager,
    }


def render_prediction_results(stored: StoredPrediction) -> None:
    """Display prediction outcome, confidence, and database save confirmation."""
    result = stored.result
    st.subheader("Prediction result")

    is_high_risk = result.prediction_label == 1
    risk_class = "risk-high" if is_high_risk else "risk-low"

    st.markdown(
        f'<p class="{risk_class}" style="font-size:1.4rem;">{result.risk_message}</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted attrition", result.prediction_text)
    with col2:
        st.metric("Confidence", f"{result.confidence_score * 100:.1f}%")
    with col3:
        st.metric("Leave probability", f"{result.attrition_probability * 100:.1f}%")

    st.progress(
        result.attrition_probability,
        text=f"Attrition probability: {result.attrition_probability * 100:.1f}%",
    )

    st.info(f"Saved to database — prediction record **#{stored.record_id}**")

    if is_high_risk:
        st.warning(
            "This employee shows a higher likelihood of leaving. "
            "Consider retention actions (career growth, workload, compensation review)."
        )
    else:
        st.success(
            "This employee shows a lower likelihood of leaving based on the current profile."
        )

    with st.expander("Technical details"):
        st.write(format_prediction_summary(result))
        st.write(f"Database record id: {stored.record_id}")
        st.json(result.to_dict())


def render_prediction_history_dashboard() -> None:
    """
    Dashboard tab: summary metrics, charts, and prediction table from SQLite.

    Data source: database/employees.db → prediction_history table
    """
    st.subheader("Prediction history dashboard")
    st.caption("Data is loaded from SQLite table `prediction_history` (newest first).")

    col_refresh, col_limit = st.columns([1, 2])
    with col_limit:
        row_limit = st.slider(
            "Rows to display",
            min_value=5,
            max_value=200,
            value=25,
            step=5,
        )
    with col_refresh:
        st.write("")
        refresh = st.button("Refresh dashboard", use_container_width=True)

    if refresh:
        st.rerun()

    try:
        metrics = get_summary_metrics()
        history_table = get_history_table(limit=row_limit)
        outcome_counts = get_outcome_counts()
        confidence_by_outcome = get_confidence_by_outcome()
    except DatabaseError as exc:
        st.error(f"Could not load prediction history: {exc}")
        st.code("python scripts/init_database.py", language="bash")
        return

    # --- Summary metrics ---
    st.markdown("### Summary statistics")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total predictions", metrics["total_predictions"])
    with m2:
        st.metric("High risk (Yes)", metrics["attrition_predictions"])
    with m3:
        st.metric("Low risk (No)", metrics["stay_predictions"])
    with m4:
        st.metric("Avg confidence", f"{metrics['avg_confidence']}%")

    if metrics["total_predictions"] == 0:
        st.info(
            "No predictions stored yet. Use the **New prediction** tab "
            "and click **Predict Attrition** to create history records."
        )
        return

    st.caption(f"Attrition rate (high risk): **{metrics['attrition_rate_pct']}%** of stored predictions")

    # --- Charts ---
    st.markdown("### Charts")
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown("**Outcomes**")
        st.bar_chart(outcome_counts.set_index("Outcome"))

    with chart_right:
        st.markdown("**Average confidence by outcome**")
        if not confidence_by_outcome.empty:
            chart_df = confidence_by_outcome.rename(
                columns={
                    "prediction_label": "Outcome",
                    "avg_confidence": "Avg confidence (0-1)",
                }
            )
            st.bar_chart(chart_df.set_index("Outcome"))
        else:
            st.caption("Not enough data for confidence chart.")

    # --- Table ---
    st.markdown("### Recent predictions")
    st.dataframe(history_table, use_container_width=True, hide_index=True)

    with st.expander("How does this dashboard load data?"):
        st.markdown(
            """
            1. `fetch_prediction_summary()` runs SQL aggregates on `prediction_history`.  
            2. `fetch_prediction_history()` selects recent rows and parses `employee_input` JSON.  
            3. This page formats the result for metrics, charts, and `st.dataframe()`.  
            """
        )


def render_prediction_page() -> None:
    """New prediction tab: employee form and live results."""
    employee_input = render_employee_form()

    if employee_input is None:
        st.info("Fill in the form and click **Predict Attrition** to see results.")
        if "last_stored_prediction" in st.session_state:
            st.divider()
            st.caption("Last prediction (saved in session and database):")
            render_prediction_results(st.session_state["last_stored_prediction"])
        return

    with st.spinner("Running prediction and saving to database..."):
        try:
            stored = predict_and_store(employee_input)
            st.session_state["last_stored_prediction"] = stored
            render_prediction_results(stored)
        except PredictionServiceError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


def main() -> None:
    """Application entry point."""
    render_sidebar()

    st.title("HR Attrition Prediction")
    st.markdown(
        "Predict whether an employee is likely to leave the company using a "
        "**Logistic Regression** model trained on historical HR data."
    )

    ready, missing = artifacts_ready()
    if not ready:
        st.error("Model artifacts are missing. Train the model before predicting.")
        st.code(
            "python scripts/init_database.py\npython scripts/train_model.py\nstreamlit run app.py",
            language="bash",
        )
        for path in missing:
            st.caption(f"Missing file: {path}")
        return

    tab_predict, tab_history = st.tabs(["New prediction", "Prediction history"])

    with tab_predict:
        render_prediction_page()

    with tab_history:
        render_prediction_history_dashboard()


if __name__ == "__main__":
    main()

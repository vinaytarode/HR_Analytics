-- HR Attrition Prediction — SQLite schema (Phase 2)
-- Database file: database/employees.db

-- ---------------------------------------------------------------------------
-- employee_data
-- Stores the full HR dataset imported from data/HR_Attrition.csv
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS employee_data (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    Age                     INTEGER NOT NULL,
    Attrition               TEXT    NOT NULL,
    BusinessTravel          TEXT    NOT NULL,
    DailyRate               INTEGER NOT NULL,
    Department              TEXT    NOT NULL,
    DistanceFromHome        INTEGER NOT NULL,
    Education               INTEGER NOT NULL,
    EducationField          TEXT    NOT NULL,
    EmployeeCount           INTEGER NOT NULL,
    EmployeeNumber          INTEGER NOT NULL UNIQUE,
    EnvironmentSatisfaction INTEGER NOT NULL,
    Gender                  TEXT    NOT NULL,
    HourlyRate              INTEGER NOT NULL,
    JobInvolvement          INTEGER NOT NULL,
    JobLevel                INTEGER NOT NULL,
    JobRole                 TEXT    NOT NULL,
    JobSatisfaction         INTEGER NOT NULL,
    MaritalStatus           TEXT    NOT NULL,
    MonthlyIncome           INTEGER NOT NULL,
    MonthlyRate             INTEGER NOT NULL,
    NumCompaniesWorked      INTEGER NOT NULL,
    Over18                  TEXT    NOT NULL,
    OverTime                TEXT    NOT NULL,
    PercentSalaryHike       INTEGER NOT NULL,
    PerformanceRating       INTEGER NOT NULL,
    RelationshipSatisfaction INTEGER NOT NULL,
    StandardHours           INTEGER NOT NULL,
    StockOptionLevel        INTEGER NOT NULL,
    TotalWorkingYears       INTEGER NOT NULL,
    TrainingTimesLastYear   INTEGER NOT NULL,
    WorkLifeBalance         INTEGER NOT NULL,
    YearsAtCompany          INTEGER NOT NULL,
    YearsInCurrentRole      INTEGER NOT NULL,
    YearsSinceLastPromotion INTEGER NOT NULL,
    YearsWithCurrManager    INTEGER NOT NULL,
    imported_at             TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------------------
-- prediction_history
-- Stores each Streamlit prediction for audit and dashboard (Phase 7–8)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL DEFAULT (datetime('now')),
    prediction_label  TEXT    NOT NULL,
    confidence_score  REAL    NOT NULL,
    risk_message      TEXT    NOT NULL,
    employee_input    TEXT    NOT NULL  -- JSON string of form fields
);

CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at
    ON prediction_history (created_at DESC);

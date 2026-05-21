# Setup and run guide (Phase 9)

Complete instructions to run the HR Attrition Prediction application locally.

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python 3.10+** | Check with `python3 --version` |
| **pip** | Usually included with Python |
| **Git** (optional) | To clone the repository |

Built-in modules used (no extra install): `sqlite3`, `json`, `pathlib`, `logging`.

---

## 1. Open the project folder

```bash
cd /home/tarodevinay/hr_project_env/hr_attrition_project
```

All commands below assume you are in this directory.

---

## 2. Create a virtual environment (recommended)

A virtual environment keeps dependencies isolated from your system Python.

```bash
python3 -m venv venv
```

Activate it:

**Linux / macOS:**

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

Your prompt should show `(venv)` when active.

Deactivate later with:

```bash
deactivate
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### What each package does

| Package | Purpose |
|---------|---------|
| `pandas` | Load CSV / SQLite data as DataFrames |
| `numpy` | Numeric arrays for ML input |
| `scikit-learn` | Logistic Regression, StandardScaler, metrics |
| `joblib` | Save/load `.pkl` model files |
| `streamlit` | Web UI (`streamlit run app.py`) |

---

## 4. Initialize the database (one-time)

Imports `data/HR_Attrition.csv` into `database/employees.db`.

```bash
python scripts/init_database.py
```

Expected output:

```
Imported 1470 employee rows ...
employee_data row count: 1470
```

---

## 5. Train the model (one-time, or after retraining)

Creates `models/attrition_model.pkl`, `scaler.pkl`, and metadata files.

```bash
python scripts/train_model.py
```

Expected output:

```
Test accuracy: 86.05%
```

---

## 6. Run the Streamlit application

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually **http://localhost:8501**).

### App tabs

1. **New prediction** — enter employee details and predict  
2. **Prediction history** — view saved predictions from SQLite  

---

## Quick setup (automated script)

Runs steps 2–5 automatically:

```bash
bash scripts/setup_project.sh
source venv/bin/activate
streamlit run app.py
```

---

## Run all tests

After setup, verify each layer:

```bash
bash scripts/run_tests.sh
```

Or run individually:

```bash
python scripts/test_preprocess.py
python scripts/test_predictor.py
python scripts/test_prediction_integration.py
python scripts/test_history_dashboard.py
```

---

## Daily development workflow

```bash
cd hr_attrition_project
source venv/bin/activate
streamlit run app.py
```

Only repeat **init_database** or **train_model** when:

- CSV data changes  
- You delete `database/employees.db`  
- You delete `models/*.pkl`  

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model file not found` | Run `python scripts/train_model.py` |
| `employee_data table is empty` | Run `python scripts/init_database.py` |
| `ModuleNotFoundError: utils` | Run commands from `hr_attrition_project/` or use scripts in `scripts/` |
| Streamlit port in use | `streamlit run app.py --server.port 8502` |
| Wrong Python packages | Activate `venv` before `pip install` |

---

## File checklist after setup

```
hr_attrition_project/
├── database/employees.db          ✓ created by init_database.py
├── models/attrition_model.pkl       ✓ created by train_model.py
├── models/scaler.pkl                ✓
├── models/feature_columns.json      ✓
├── models/encoding_metadata.json    ✓
└── venv/                            ✓ optional but recommended
```

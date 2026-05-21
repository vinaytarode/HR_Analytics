# Phase 10 — Architecture review

Final review of the HR Attrition Prediction application (all phases complete).

**Verification date:** All `scripts/run_tests.sh` tests passed.

---

## 1. Architecture summary

```
┌─────────────────────────────────────────────────────────────┐
│  app.py (Streamlit)                                         │
│    Tab 1: New prediction    Tab 2: History dashboard        │
└────────────┬───────────────────────────────┬────────────────┘
             │                               │
             ▼                               ▼
┌────────────────────────┐      ┌────────────────────────────┐
│ prediction_service.py  │      │ history_dashboard.py       │
│  predict_and_store()   │      │  get_summary_metrics()     │
└──────────┬─────────────┘      └─────────────┬──────────────┘
           │                                  │
     ┌─────┴─────┐                            │
     ▼           ▼                            ▼
predictor.py  database.py ◄────────────────────┘
     │           │
     ▼           ▼
preprocess.py  employees.db
     │
     ▼
train_model.py → models/*.pkl
```

**Design strengths**

- Clear separation: UI / service / ML / data  
- Single preprocessing module (`preprocess.py`) for train + infer  
- Paths anchored to project root (not cwd-dependent)  
- SQLite for persistence without a database server  

---

## 2. Component checklist

| Area | Status | Notes |
|------|--------|-------|
| Imports | ✅ | All `utils.*` imports resolve when run from project root |
| File paths | ✅ | `PROJECT_ROOT` pattern consistent in `database`, `preprocess`, `train_model`, `app` |
| Database | ✅ | `get_db_connection`, `BEGIN IMMEDIATE` on writes, rollback on error |
| Preprocessing | ✅ | Batch vs single-row test passes (`test_preprocess.py`) |
| Model loading | ✅ | `joblib` load via `load_model()` / `load_scaler()` |
| Streamlit | ✅ | Form → predict → save → history tab |
| Error handling | ✅ | `DatabaseError`, `PredictionError`, `PredictionServiceError` |

---

## 3. Data flow (verified)

1. **Bootstrap:** `HR_Attrition.csv` → `init_database.py` → `employee_data` (1470 rows)  
2. **Training:** SQLite → `preprocess_for_training()` → LR + scaler → `.pkl` + JSON metadata  
3. **Inference:** Form dict → `preprocess_for_prediction()` → scale → predict  
4. **Persistence:** `insert_prediction()` → `prediction_history`  
5. **Dashboard:** SQL fetch → format → `st.dataframe` / charts  

---

## 4. Common bugs and prevention

| Bug | Prevention in this project |
|-----|---------------------------|
| Train/infer preprocessing mismatch | Shared `utils/preprocess.py`; `test_preprocess.py` |
| Wrong feature column order | `feature_columns.json` + `align_feature_columns()` |
| One-hot failure on single row | `apply_one_hot_encoding_inference()` |
| Missing model files | `artifacts_ready()` check in `app.py` |
| Empty database | `init_database.py` + error messages |
| Retraining on every request | Saved `.pkl` artifacts |
| SQLite locked | `timeout=30`, `BEGIN IMMEDIATE` |

---

## 5. Debugging strategy

1. **Layer tests** (bottom-up):  
   `test_preprocess.py` → `test_predictor.py` → `test_prediction_integration.py` → `test_history_dashboard.py`

2. **SQLite inspection:**  
   `sqlite3 database/employees.db ".tables"`  
   `SELECT * FROM prediction_history ORDER BY id DESC LIMIT 5;`

3. **Artifacts:** Confirm `models/*.pkl` and JSON files exist after `train_model.py`.

4. **Logging:** Predictor/service modules use `logging.INFO` — run CLI tests to see output.

5. **Streamlit:** Use expander “Technical details” and sidebar “Load employee” to reproduce known rows.

---

## 6. Scalability limitations

| Limitation | Why | Typical fix |
|------------|-----|-------------|
| SQLite single-writer | Not ideal for many concurrent writes | PostgreSQL / cloud DB |
| Streamlit rerun model | Whole script reruns per interaction | FastAPI + React for scale |
| In-memory model per process | Each worker loads `.pkl` | Model server (MLflow, BentoML) |
| No authentication | Local demo only | Auth layer (OAuth, API keys) |
| Batch training only | No online learning | Scheduled retraining pipeline |
| Class imbalance | Attrition ~16%; LR may under-predict minority | SMOTE, class weights, threshold tuning |

**Appropriate for:** local demos, portfolios, small HR teams, proof-of-concept.

**Not appropriate for:** high-traffic multi-tenant SaaS without architectural changes.

---

## 7. Future improvements

- **API layer:** FastAPI endpoint wrapping `predict_and_store()`  
- **Auth & roles:** HR admin vs read-only viewer  
- **Model versioning:** Track `training_report.json` per deploy  
- **Explainability:** SHAP values for feature importance in UI  
- **Docker:** `Dockerfile` + `docker-compose` for one-command deploy  
- **CI/CD:** GitHub Actions running `run_tests.sh` on push  
- **Threshold tuning:** Adjust decision boundary for precision/recall trade-off  
- **Export:** Download prediction history as CSV from dashboard  

---

## 8. Phase completion

All 10 phases delivered. Application is fully runnable per `SETUP.md`.

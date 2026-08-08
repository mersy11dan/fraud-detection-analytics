# Fraud Detection Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-pytest-green?logo=pytest&logoColor=white)](tests/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey?logo=github)](.github/workflows/ci.yml)

End-to-end machine learning project for **unified fraud detection across e-commerce and banking transaction streams**. The pipeline covers data cleaning, EDA, feature engineering, class imbalance handling, tuned classifiers (LR / RF / XGBoost), SHAP explainability, executive insights, and a Streamlit dashboard.

---

## Project Overview

This repository implements a **unified fraud analytics workflow** across two transaction streams:

- **Fraud_Data** — e-commerce checkout fraud (behavioral, channel, device, geolocation features)
- **creditcard.csv** — banking / card fraud (PCA features + amount/time)

Both streams share the same modeling stack (stratified split → SMOTE on train → tune LR/RF/XGBoost on PR-AUC → holdout evaluation → SHAP). The project is a modular Python package (`src/`) with CLI scripts, notebooks, tests, Streamlit dashboard, and report-ready artifacts under `reports/`.

---

## Business Problem

Online merchants lose revenue and customer trust when fraudulent transactions are approved. Manual review of every order is not scalable; fully automated blocking risks false declines and poor customer experience.

**Objective:** Build a **unified risk-scoring framework** that prioritizes suspicious transactions for review across both e-commerce checkout and banking/card streams, keeping false alarms low. Success is measured by precision-recall trade-offs appropriate for a **review-first** deployment — high precision to protect analyst capacity, with stream-specific features and thresholds.

---

## Dataset Description

| Dataset | File | Rows (approx.) | Fraud rate | Role |
|---------|------|----------------|------------|------|
| E-commerce fraud | `Fraud_Data.csv` | 151,112 | 9.36% | Primary modeling dataset |
| Credit card (PCA features) | `creditcard.csv` | 284,807 (283,726 clean) | 0.17% | Full modeling + SHAP |
| IP geolocation | `IpAddress_to_Country.csv` | 138,846 | — | Country enrichment lookup |

**Fraud_Data key fields:** `user_id`, signup/purchase timestamps, `purchase_value`, `device_id`, acquisition `source`, `browser`, `sex`, `age`, `ip_address`, and binary `class` (fraud label).

Place raw CSVs in `data/raw/` before running pipelines. Processed outputs are written to `data/processed/` (gitignored).

---

## Project Structure

```
fraud-detection-analytics/
├── dashboard/                 # Streamlit app (app.py)
├── data/
│   ├── raw/                   # Source CSVs (gitignored)
│   └── processed/             # Pipeline outputs (gitignored)
├── docs/                      # Interim report (HTML/Markdown)
├── notebooks/
│   ├── eda-fraud-data.ipynb   # Exploratory analysis
│   └── modeling.ipynb         # Model training & evaluation
├── reports/                   # Generated diagnostics & insights
│   ├── modeling/              # Metrics, feature importance (legacy path)
│   ├── outputs/               # Model comparison CSVs, plots (primary)
│   ├── shap/                  # SHAP summaries
│   └── fraud_insights.md      # Executive recommendations
├── scripts/                   # CLI entry points
├── src/
│   ├── data/                  # Dataset loaders
│   ├── features/              # Feature engineering
│   ├── modeling/              # Training, tuning, SHAP, insights
│   ├── preprocessing/         # Cleaning, geolocation, inspection
│   └── utils/
├── tests/                     # pytest suite
├── requirements.txt
└── README.md
```

---

## Methodology

### Data Cleaning

- Inspect schema, dtypes, missing values, and duplicates for both datasets.
- Parse timestamps; remove duplicate rows from `creditcard.csv` (1,081 removed).
- Standardize column names and validate fraud label distributions.
- Entry point: `python scripts/run_preprocess.py`

### EDA

- Profile fraud rate by acquisition channel, geography, and purchase timing.
- Key finding: median signup-to-purchase time is **~0 hours** for fraud vs **~1,443 hours** for legitimate users.
- Direct traffic shows the highest channel fraud rate (~10.5%).
- Notebook: `notebooks/eda-fraud-data.ipynb`

### Feature Engineering

- **Temporal:** `time_since_signup_hours`, `hour_of_day`, `day_of_week`
- **Velocity:** rolling transaction counts and user velocity (limited signal — one txn/user in current data)
- **Geolocation:** IP-to-country mapping (85.5% match rate)
- **Encoding:** scaled numerics + one-hot categoricals → 203 features + label
- Entry points: `run_geolocation_enrichment.py`, `run_feature_engineering.py`

### Class Imbalance Handling

- Stratified **80/20** train/test split (`RANDOM_STATE=42`); holdout retains ~9.4% fraud prevalence.
- **SMOTE** applied to the training split only — test set never resampled.
- Training distribution balanced from 9.36% → 50% fraud after SMOTE.
- Entry point: `python scripts/run_imbalance_resampling.py`

### Modeling

| Model | Tuning strategy | CV metric |
|-------|-----------------|-----------|
| Logistic Regression | `GridSearchCV` (`C`) | PR-AUC |
| Random Forest | `RandomizedSearchCV` | PR-AUC |
| XGBoost | `RandomizedSearchCV` | PR-AUC |

- 3-fold cross-validation on the pre-SMOTE training split; final models refit on SMOTE-balanced data.
- Metrics: precision, recall, F1, ROC-AUC, **PR-AUC** (primary selection metric).
- Best model selected automatically by holdout PR-AUC.
- Entry points: `python scripts/run_modeling_reports.py`, `notebooks/modeling.ipynb`

### Explainability

- Feature importance from the best model (Random Forest).
- **SHAP** analysis: summary, bar, waterfall, and dependence plots.
- Stakeholder markdown summaries for fraud analysts and executives.
- Entry points: `python scripts/run_shap_analysis.py`, `python scripts/run_fraud_insights.py`

---

## Results

### E-commerce (`Fraud_Data`) — best: `random_forest_tuned`

| Metric | Value |
|--------|-------|
| PR-AUC | 0.625 |
| Precision | 99.1% |
| Recall | 52.7% |
| F1 | 0.688 |
| ROC-AUC | 0.770 |

Holdout (30,223): TP 1,492 · FP 13 · FN 1,338 · TN 27,380  
**Top driver:** `time_since_signup_hours` (~69% importance)

### Banking (`creditcard.csv`) — best: `random_forest`

| Metric | Value |
|--------|-------|
| PR-AUC | 0.809 |
| Precision | 92.3% |
| Recall | 75.8% |
| F1 | 0.832 |
| ROC-AUC | 0.975 |

Holdout (56,746): TP 72 · FP 6 · FN 23 · TN 56,645  
**Top drivers:** PCA components `V14`, `V17`, `V10`, `V12`, `V16`

Artifacts: `reports/outputs/` · `reports/outputs/creditcard/` · `reports/shap/` · `reports/shap/creditcard/`

---

## Key Insights

1. **Unified two-stream solution** — the same stack scores e-commerce checkout fraud and banking/card fraud with stream-specific features and thresholds.
2. **Signup velocity dominates e-commerce** — fraudsters buy almost immediately after signup; legitimate users wait weeks.
3. **PCA components dominate banking** — `V14`, `V17`, `V10` lead creditcard importance/SHAP; amount alone is weak.
4. **High precision supports review-first deployment** — 13 false positives (e-commerce) and 6 (creditcard) on holdout.
5. **Prevalence drives metric choice** — PR-AUC is the selection metric for both streams; accuracy is misleading on creditcard (~0.17% fraud).
6. **Rules + ML** — new-account cooldowns help e-commerce; card streams rely more on model scores because features are anonymized.

---

## Dashboard

An interactive **Streamlit** dashboard visualizes KPIs, EDA, model performance, SHAP drivers, and executive recommendations.

```bash
streamlit run dashboard/app.py
```

**Pages:** Overview · Dataset Summary · Fraud EDA · Model Performance · SHAP Explainability · Executive Recommendations

The dashboard loads artifacts from `reports/` and processed data from `data/processed/`.

---

## Portfolio website

A static portfolio case study lives at [`docs/index.html`](docs/index.html).

**Live (after enabling GitHub Pages):**  
`https://mersy11dan.github.io/fraud-detection-analytics/`

**Local preview:** open `docs/index.html` in a browser, or:

```bash
python -m http.server 8080 --directory docs
```

Then visit `http://localhost:8080`.

### Deploy on GitHub Pages

1. Push this repo (branch `task-2` or `main`).
2. GitHub → **Settings** → **Pages**.
3. Under **Build and deployment**:
   - Source: **GitHub Actions** (recommended — uses `.github/workflows/pages.yml`), **or**
   - Source: **Deploy from a branch** → Branch `task-2` (or `main`) → Folder **`/docs`** → Save.
4. Wait 1–2 minutes, then open:  
   `https://mersy11dan.github.io/fraud-detection-analytics/`
5. The full report is at:  
   `https://mersy11dan.github.io/fraud-detection-analytics/final-report.html`

---

## Documentation

| Document | Path |
|----------|------|
| Portfolio site | [`docs/index.html`](docs/index.html) |
| Final report (Markdown) | [`docs/final-report.md`](docs/final-report.md) |
| Final report (PDF export) | [`docs/final-report.html`](docs/final-report.html) |
| Interim report | [`docs/week-5-6-interim-report.md`](docs/week-5-6-interim-report.md) |
| Final audit checklist | [`reports/final_checklist.md`](reports/final_checklist.md) |

Regenerate HTML: `python scripts/build_final_report_html.py`

---

## Installation

**Prerequisites:** Python 3.10+, Git

```bash
git clone <repository-url>
cd fraud-detection-analytics

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Place raw datasets in `data/raw/`:

- `Fraud_Data.csv`
- `creditcard.csv`
- `IpAddress_to_Country.csv`

---

## Reproducibility

Run the full pipeline from the project root:

```bash
# Task 1 — data preparation
python scripts/run_preprocess.py
python scripts/run_geolocation_enrichment.py
python scripts/run_feature_engineering.py
python scripts/run_imbalance_resampling.py

# Task 2 — unified modeling (both streams)
python scripts/run_unified_modeling.py
# or separately:
python scripts/run_modeling_reports.py
python scripts/run_creditcard_modeling.py

# Explainability (both streams)
python scripts/run_shap_analysis.py
python scripts/run_creditcard_shap.py
python scripts/run_fraud_insights.py

# Reports
python scripts/build_final_report_html.py
python scripts/generate_final_report.py

# Dashboard
streamlit run dashboard/app.py
```

**Verify with tests:**

```bash
pytest tests/ -v
```

CI runs the same test suite on push/PR via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

**Configuration:** `src/config.py` — `RANDOM_STATE=42`, `TEST_SIZE=0.2`, output paths.

**Python API example:**

```python
from src.modeling import run_fraud_modeling_workflow, run_shap_explainability_workflow

workflow = run_fraud_modeling_workflow(save_report=True)
shap_analysis = run_shap_explainability_workflow(workflow=workflow)
```

---

## Future Improvements

- **Threshold tuning** — optimize decision thresholds on the PR curve per stream for cost-sensitive targets.
- **Resampling comparison** — benchmark partial SMOTE against class weights, ADASYN, and undersampling (especially creditcard).
- **Velocity features** — re-engineer when e-commerce repeat-purchase data is available.
- **Model monitoring** — feature drift (signup timing / top PCA components) and scheduled retraining.
- **Production deployment** — persist models and expose a dual-stream scoring API.

---

## License

Add license details here if required by your course or organization.

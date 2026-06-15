# Fraud Detection Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/tests-pytest-green?logo=pytest&logoColor=white)](tests/)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey?logo=github)](.github/workflows/ci.yml)

End-to-end machine learning project for detecting fraudulent e-commerce transactions. The pipeline covers data cleaning, exploratory analysis, feature engineering, class imbalance handling, tuned classification models, SHAP explainability, executive insights, and an interactive Streamlit dashboard.

---

## Project Overview

This repository implements a reproducible fraud analytics workflow on real-world transaction data. The primary modeling target is **Fraud_Data** — an e-commerce dataset with user, channel, device, and behavioral attributes. A secondary **creditcard.csv** dataset is preprocessed for future modeling.

The project is structured as a modular Python package (`src/`) with CLI scripts, Jupyter notebooks, automated tests, and report-ready artifacts under `reports/`. Design goals include leakage-safe preprocessing, imbalanced-learning best practices, and stakeholder-ready outputs for data scientists, fraud analysts, and executives.

---

## Business Problem

Online merchants lose revenue and customer trust when fraudulent transactions are approved. Manual review of every order is not scalable; fully automated blocking risks false declines and poor customer experience.

**Objective:** Build a model that reliably prioritizes suspicious transactions for review while keeping false alarms low. Success is measured by precision-recall trade-offs appropriate for a **review-first** deployment — high precision to protect analyst capacity, with recall improvements targeted through threshold tuning and verification workflows.

---

## Dataset Description

| Dataset | File | Rows (approx.) | Fraud rate | Role |
|---------|------|----------------|------------|------|
| E-commerce fraud | `Fraud_Data.csv` | 151,112 | 9.36% | Primary modeling dataset |
| Credit card (PCA features) | `creditcard.csv` | 284,807 | 0.17% | Preprocessed; modeling planned |
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

**Best model:** `random_forest_tuned` (selected by PR-AUC)

| Metric | Value |
|--------|-------|
| PR-AUC | 0.625 |
| Precision | 99.1% |
| Recall | 52.7% |
| F1 | 0.688 |
| ROC-AUC | 0.770 |

**Holdout confusion matrix (30,223 transactions):**

| | Predicted Legitimate | Predicted Fraud |
|--|---------------------|-----------------|
| **Actual Legitimate** | 27,380 (TN) | 13 (FP) |
| **Actual Fraud** | 1,338 (FN) | 1,492 (TP) |

**Top feature drivers:** `time_since_signup_hours` (~69% importance), followed by geography, purchase timing, age, and purchase value.

Full artifacts: `reports/modeling/` or `reports/outputs/` · `reports/shap/` · `reports/fraud_insights.md`

---

## Key Insights

1. **Signup velocity is the dominant signal** — fraudsters purchase almost immediately after account creation; legitimate customers wait weeks on average.
2. **High precision supports review-first deployment** — only 13 false positives on holdout; flagged cases are strong manual-review candidates.
3. **Recall gap remains** — ~47% of fraud is missed at the default threshold; threshold tuning and verification workflows are the highest-impact next steps.
4. **Geography and channel add context** — country and browser features rank in the top 10 but should inform triage, not blanket blocking.
5. **Rules + ML combination** — new-account purchase cooldowns address the strongest signal without model inference alone.

---

## Dashboard

An interactive **Streamlit** dashboard visualizes KPIs, EDA, model performance, SHAP drivers, and executive recommendations.

```bash
streamlit run dashboard/app.py
```

**Pages:** Overview · Dataset Summary · Fraud EDA · Model Performance · SHAP Explainability · Executive Recommendations

The dashboard loads artifacts from `reports/` and processed data from `data/processed/`.

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

# Task 2 — modeling & insights
python scripts/run_modeling_reports.py
python scripts/run_shap_analysis.py
python scripts/run_fraud_insights.py

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

- **Threshold tuning** — optimize decision thresholds on the PR curve for business-specific precision/recall targets.
- **creditcard.csv modeling** — extend the pipeline to the highly imbalanced credit-card dataset.
- **Resampling comparison** — benchmark SMOTE against class weights, ADASYN, and undersampling.
- **Velocity features** — re-engineer when repeat-purchase data becomes available.
- **Model monitoring** — feature drift detection and scheduled retraining after fraud incidents.
- **Production deployment** — API scoring service and integration with case-management tooling.

---

## License

Add license details here if required by your course or organization.

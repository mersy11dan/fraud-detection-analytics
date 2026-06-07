# Fraud Detection Analytics

Week 5-6 interim submission: end-to-end fraud detection analytics with preprocessing, EDA, class imbalance handling, and model interpretability.

## Overview

<!-- Brief project summary: datasets, business problem, and modeling goals -->

This project analyzes transaction fraud using machine learning. The interim deliverable covers data ingestion, exploratory analysis, preprocessing pipelines, and baseline modeling for highly imbalanced fraud labels.

## Setup

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running tests

```bash
pytest tests/ -v
```

## Data

<!-- Describe datasets: creditcard.csv, Fraud_Data.csv, IpAddress_to_Country.csv -->

| Dataset | Location | Description |
|---------|----------|-------------|
| Credit card transactions | `data/raw/creditcard.csv` | PCA-transformed features with `Class` fraud label |
| E-commerce fraud | `data/raw/Fraud_Data.csv` | User/session transaction features |
| IP geolocation | `data/raw/IpAddress_to_Country.csv` | IP-to-country mapping for enrichment |

Raw data is excluded from version control. Place files in `data/raw/` before running notebooks or scripts.

## Preprocessing

<!-- Document cleaning, feature engineering, train/test splits, and pipeline modules -->

Preprocessing logic lives in `src/preprocessing/`. Processed outputs are written to `data/processed/`.

Planned steps:

- Missing value handling and type coercion
- Feature scaling and encoding
- Temporal and geolocation enrichment
- Reproducible train/validation/test splits

## EDA

<!-- Summarize key exploratory findings: class distribution, amount patterns, time trends -->

Exploratory analysis notebooks are in `notebooks/`. Focus areas:

- Fraud rate and class imbalance
- Transaction amount and time distributions
- Correlation and feature importance previews
- Segment-level fraud patterns

## Class Imbalance

<!-- Describe resampling strategy: SMOTE, class weights, evaluation metrics -->

Fraud detection is severely imbalanced. The project uses `imbalanced-learn` for resampling and evaluates models with metrics suited to rare events (e.g., precision-recall AUC, F1, recall at fixed precision).

## Interim Submission

<!-- Checklist of completed interim deliverables -->

**Deliverables for Week 5-6:**

- [ ] Project structure and reproducible environment (`requirements.txt`, CI workflow)
- [ ] Data loading and preprocessing modules (`src/`)
- [ ] EDA notebooks with documented insights
- [ ] Baseline model training and evaluation
- [ ] Class imbalance handling experiments
- [ ] README updates with findings and next steps

## Project Structure

```
fraud-detection-analytics/
├── .github/workflows/   # CI pipeline
├── data/
│   ├── raw/             # Source datasets (gitignored)
│   └── processed/       # Cleaned feature tables (gitignored)
├── models/              # Trained model artifacts (gitignored)
├── notebooks/           # EDA and experimentation
├── scripts/             # CLI entry points
├── src/                 # Reusable Python modules
└── tests/               # Unit tests
```

## License

<!-- Add license if required by course -->

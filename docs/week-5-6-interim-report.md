# Fraud Detection Analytics — Week 5–6 Interim Report (Task 1)

**Project:** fraud-detection-analytics  
**Scope:** Data understanding, preprocessing, EDA, geolocation enrichment, feature engineering, and class imbalance handling  
**Author:** Interim submission  
**Date:** June 2026

---

## 1. Project Overview

This interim deliverable establishes the foundation for a fraud detection analytics pipeline using two complementary datasets: e-commerce transaction logs (`Fraud_Data.csv`) and credit card transactions (`creditcard.csv`). A third reference table (`IpAddress_to_Country.csv`) supports geolocation enrichment.

Task 1 focuses on making the work **reproducible and modular**. The repository now includes:

- A structured Python package under `src/` for loading, cleaning, enrichment, and feature engineering
- An EDA notebook for `Fraud_Data` (`notebooks/eda-fraud-data.ipynb`)
- CLI scripts for preprocessing, geolocation, feature engineering, and imbalance handling
- A GitHub Actions workflow that installs dependencies and runs **43 pytest tests**
- Processed outputs saved to `data/processed/` and diagnostic reports to `reports/`

The immediate goal is not final model deployment, but a reliable path from raw data to model-ready features with documented data quality and class imbalance decisions.

---

## 2. Data Understanding

### Fraud_Data.csv (e-commerce)

| Attribute | Value |
|-----------|-------|
| Rows | 151,112 |
| Columns | 11 |
| Target | `class` (0 = legitimate, 1 = fraud) |
| Key fields | `signup_time`, `purchase_time`, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address` |

After loading, the raw file contains **no missing values** and **no duplicate rows**. Fraud accounts for **9.36%** of transactions (14,151 fraud / 136,961 legitimate), yielding an imbalance ratio of approximately **9.7:1**.

Each `user_id` appears exactly once in this dataset, meaning every row represents a single user transaction.

### creditcard.csv (PCA-transformed features)

| Attribute | Value |
|-----------|-------|
| Raw rows | 284,807 |
| Columns | 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Clean rows (after preprocessing) | 283,726 |

This dataset is **severely imbalanced**: only **0.17%** of transactions are fraudulent (473 fraud / 283,253 legitimate), with a ratio near **599:1**. Features are already PCA components, so direct feature interpretation is limited. `Time` records seconds elapsed since the first transaction in the dataset.

### IpAddress_to_Country.csv

| Attribute | Value |
|-----------|-------|
| Rows | 138,846 |
| Columns | `lower_bound_ip_address`, `upper_bound_ip_address`, `country` |

This file maps IPv4 ranges to country names and is used for range-based IP enrichment on `Fraud_Data`.

---

## 3. Cleaning and Preprocessing

Preprocessing is implemented in reusable modules (`src/preprocessing/`) and applied consistently across scripts and notebooks.

**Fraud_Data pipeline**

1. Standardize column names to lowercase snake_case
2. Parse `signup_time` and `purchase_time` as datetimes
3. Remove duplicate rows (none found in raw data)
4. Handle missing values (none found; pipeline still enforces required columns)
5. Coerce numeric and string dtypes for modeling

**creditcard.csv pipeline**

1. Standardize column names
2. Remove **1,081 duplicate rows**
3. Cast `class`, `amount`, `time`, and PCA features to numeric types

All cleaning steps log actions to stdout, which supports auditability during notebook and script runs. Unit tests verify missing-value handling, duplicate removal, and timestamp conversion on synthetic data.

---

## 4. EDA Findings

### Fraud_Data

Exploratory analysis in `notebooks/eda-fraud-data.ipynb` highlights behavioral and channel-level patterns.

**Class imbalance**

Fraud is the minority class but not negligible at ~9.4%. Accuracy alone would be a poor evaluation metric; precision, recall, F1, and PR-AUC are more appropriate.

![Fraud_Data class distribution](figures/fraud_class_distribution.png)

**Signup-to-purchase timing is the strongest signal**

Legitimate users have a median time from signup to purchase of approximately **1,443 hours** (~60 days). Fraudulent users have a median near **0 hours** — essentially immediate purchases after signup. This motivates `time_since_signup_hours` as a primary engineered feature.

![Signup-to-purchase time by class](figures/time_since_signup_by_class.png)

*Note: The boxplot clips at the 95th percentile for readability; fraud remains near zero even without clipping.*

**Channel and demographic patterns (modest but useful)**

| Segment | Fraud rate |
|---------|------------|
| Direct traffic | 10.54% |
| Ads | 9.21% |
| SEO | 8.93% |
| Chrome | 9.88% |
| Firefox | 9.52% |
| Safari | 9.02% |
| Male | 9.55% |
| Female | 9.10% |

Median `purchase_value` (~35) and median `age` (~33) are similar across classes, so amount and age alone do not separate fraud well.

### creditcard.csv

Preprocessing and descriptive review show a fundamentally different problem shape:

| Metric | Legitimate (0) | Fraud (1) |
|--------|----------------|-----------|
| Share of rows | 99.83% | 0.17% |
| Median `amount` | €22.00 | €9.82 |
| Median `time` (seconds) | 84,711 | 73,408 |

Fraud transactions tend to be slightly lower in amount and occur earlier in the dataset timeline. The extreme rarity of fraud cases (under 500 rows) will require careful resampling and metric selection in later modeling work.

![Credit card class distribution](figures/creditcard_class_distribution.png)

*Y-axis uses a log scale because the majority class dominates.*

---

## 5. Geolocation Enrichment

IP addresses in `Fraud_Data` are stored as numeric values. The enrichment pipeline (`src/preprocessing/geolocation.py`):

1. Converts IPs to 32-bit integers (`ip_address_int`)
2. Performs a **range-based lookup** against `IpAddress_to_Country.csv` using `pd.merge_asof`
3. Validates that each IP falls within the matched range upper bound
4. Assigns `country`, or `"unknown"` when no range matches

**Results (from `scripts/run_geolocation_enrichment.py`)**

| Metric | Value |
|--------|-------|
| Transactions enriched | 151,112 |
| IPs matched to a country | 129,146 (**85.46%**) |
| Unmatched (`unknown`) | 21,966 |
| Distinct countries | 182 |

Among countries with at least 100 transactions, higher observed fraud rates included Ecuador (26.4%), Tunisia (26.3%), and Peru (26.1%). These segment-level patterns support including `country` as a categorical feature, while recognizing that high rates in smaller geographies may reflect limited sample size.

Output saved to: `data/processed/fraud_data_geolocated.csv`

---

## 6. Feature Engineering

Feature engineering is implemented in `src/features/` and executed via `scripts/run_feature_engineering.py`.

### Temporal features

| Feature | Description |
|---------|-------------|
| `time_since_signup_hours` | Hours between `signup_time` and `purchase_time` |
| `hour_of_day` | Purchase hour (0–23) |
| `day_of_week` | Purchase weekday (0 = Monday) |

`time_since_signup_hours` is the most important engineered feature given EDA: median **0.0003 h** for fraud vs **1,443 h** for legitimate users in the engineered dataset.

### Velocity features

The pipeline also computes per-user rolling transaction counts (`txn_count_last_1h`, `txn_count_last_24h`, `txn_count_last_168h`), `hours_since_last_txn`, `user_cumulative_txn_count`, and `user_txn_velocity_per_day`.

Because each `user_id` in `Fraud_Data` has only one transaction, these velocity features are uniformly **1** (or **0** hours since last transaction) in the current dataset. The code remains valuable for datasets with repeat purchasers.

### Encoding and scaling

- Numeric features are standardized with `StandardScaler`
- Categorical features (`source`, `browser`, `sex`, `country`) are one-hot encoded
- High-cardinality `device_id` is excluded from one-hot encoding to avoid sparse, unstable columns

**Outputs**

| File | Shape / size |
|------|----------------|
| `data/processed/fraud_data_engineered.csv` | 151,112 rows × 22 columns |
| `data/processed/fraud_data_features.csv` | 151,112 rows × 204 columns (203 features + `class`) |

---

## 7. Class Imbalance Handling

Fraud detection models trained on raw class proportions tend to favor the majority class. We address this with **SMOTE on the training split only**, implemented in `src/modeling/imbalance.py`.

**Why SMOTE over undersampling**

- `Fraud_Data` has ~9.4% fraud — imbalanced, but not so rare that majority rows should be discarded
- SMOTE keeps all 109,568 legitimate training rows and synthesizes additional fraud examples
- Random undersampling would remove most legitimate transactions and waste information

**Safeguards**

1. Stratified `train_test_split` (80/20) before any resampling
2. SMOTE applied only to `(X_train, y_train)`
3. Test set left untouched to preserve real-world prevalence

**Observed distributions (from `reports/class_imbalance_summary.md`)**

| Stage | Class 0 | Class 1 | Fraud % |
|-------|---------|---------|---------|
| Train (before SMOTE) | 109,568 | 11,321 | 9.36% |
| Train (after SMOTE) | 109,568 | 109,568 | 50.0% |
| Test (holdout) | 27,393 | 2,830 | 9.36% |

![Training set class distribution before and after SMOTE](figures/smote_class_distribution.png)

---

## 8. Challenges and Next Steps

### Challenges encountered

1. **Different imbalance profiles** — `Fraud_Data` (~9% fraud) and `creditcard.csv` (~0.17% fraud) require dataset-specific resampling strategies; a single approach will not fit both.
2. **IP geolocation coverage** — 14.5% of IPs did not match any country range and were labeled `unknown`, which may limit geographic signal for those rows.
3. **Single-transaction users** — velocity features do not vary in the current `Fraud_Data`, reducing their usefulness until repeat behavior is present.
4. **creditcard interpretability** — PCA features limit direct business explanations; SHAP and model-based importance will matter more at the modeling stage.

### Next steps (beyond Task 1)

1. Train baseline classifiers (e.g., logistic regression, Random Forest, XGBoost) on the SMOTE-balanced training set
2. Evaluate on the untouched test set using precision, recall, F1, and PR-AUC
3. Extend preprocessing and feature engineering to `creditcard.csv`
4. Apply SHAP for interpretability on the best-performing model
5. Compare SMOTE against class-weighted models as an alternative to synthetic oversampling

---

## Appendix: Reproducibility

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Execute pipelines
python scripts/run_preprocess.py
python scripts/run_geolocation_enrichment.py
python scripts/run_feature_engineering.py
python scripts/run_imbalance_resampling.py
```

**Key artifacts**

| Artifact | Location |
|----------|----------|
| EDA notebook | `notebooks/eda-fraud-data.ipynb` |
| Geolocated data | `data/processed/fraud_data_geolocated.csv` |
| Engineered features | `data/processed/fraud_data_engineered.csv` |
| Model-ready matrix | `data/processed/fraud_data_features.csv` |
| Imbalance report | `reports/class_imbalance_summary.md` |
| Imbalance technique note | `docs/class-imbalance-handling.md` |

---

*This report reflects outputs produced by the project codebase as of the interim submission. No modeling results are included, as Task 1 focuses on data preparation and analysis.*

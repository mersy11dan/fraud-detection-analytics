# Fraud Detection Analytics — Final Report

**Project:** fraud-detection-analytics  
**Challenge:** Unified fraud detection across e-commerce and banking transaction streams  
**Date:** August 2026

---

## 1. Executive Summary

This project delivers a **unified machine learning solution** for fraud detection across two complementary transaction streams:

| Stream | Dataset | Scale | Fraud rate | Best model | Holdout PR-AUC |
|--------|---------|-------|------------|------------|----------------|
| **E-commerce** | `Fraud_Data.csv` | 151,112 txns | 9.36% | `random_forest_tuned` | **0.625** |
| **Banking / card** | `creditcard.csv` | 283,726 clean txns | 0.17% | `random_forest` | **0.809** |

**Shared methodology:** stratified train/test split → SMOTE on train only → tune Logistic Regression, Random Forest, and XGBoost on PR-AUC → evaluate on untouched holdout → SHAP explainability → stakeholder recommendations.

**E-commerce highlights:** precision **99.1%**, recall **52.7%**, only **13** false positives on 30,223 holdout transactions. Dominant driver: **short signup-to-purchase time**.

**Banking highlights:** precision **92.3%**, recall **75.8%**, F1 **0.832**, ROC-AUC **0.975**, only **6** false positives on 56,746 holdout transactions. Dominant drivers: anonymized PCA components (**V14, V17, V10**).

**Business posture:** One review-first operating model for both streams — high-precision alerts feed analyst queues; stream-specific features and thresholds handle different prevalence and signal types.

**Deliverables:** modular Python package, CLI scripts for both streams, SHAP plots, executive insights, Streamlit dashboard, and this report.

---

## 2. Business Understanding

Fraud hits merchants and issuers through chargebacks, direct loss, and eroded trust. Fully automated blocking creates false declines; reviewing every order does not scale.

**Unified business objective:** Build a **single risk-scoring framework** that serves both:

1. **E-commerce checkout fraud** — account, device, channel, and behavioral signals  
2. **Card-not-present / banking fraud** — anonymized PCA features plus amount and timing  

**Success criteria**

| Criterion | Rationale |
|-----------|-----------|
| High precision | Protect analyst capacity on both streams |
| Acceptable recall | Capture enough fraud to justify deployment |
| Stream-aware modeling | Prevalence differs (~9% vs ~0.17%); resampling and metrics must adapt |
| Explainability | Analysts need *why* a case was flagged |
| Reproducibility | Auditable pipelines, splits, and reports |

**Deployment posture:** Prefer **review-first** routing over auto-decline. E-commerce benefits from account-timing rules; banking relies more on model scores because features are anonymized.

---

## 3. Dataset Overview

### Fraud_Data.csv (e-commerce stream)

| Attribute | Value |
|-----------|-------|
| Rows | 151,112 |
| Columns | 11 |
| Target | `class` (0 = legitimate, 1 = fraud) |
| Fraud rate | 9.36% (14,151 / 136,961) |
| Imbalance ratio | ~9.7:1 |
| Missing / duplicates | None |
| Users | One transaction per `user_id` |

**Key fields:** signup/purchase times, `purchase_value`, `device_id`, `source`, `browser`, `sex`, `age`, `ip_address`.

### creditcard.csv (banking stream)

| Attribute | Value |
|-----------|-------|
| Raw rows | 284,807 |
| Clean rows | 283,726 (1,081 duplicates removed) |
| Fraud rate | 0.17% (473 / 283,253) |
| Imbalance ratio | ~599:1 |
| Features | PCA `V1`–`V28`, `Time`, `Amount` |

Median fraudulent `amount` €9.82 vs €22.00 legitimate — amount alone is a weak separator; PCA components carry the signal.

### IpAddress_to_Country.csv (e-commerce enrichment)

138,846 IPv4 ranges used only for Fraud_Data geolocation (not applicable to creditcard PCA features).

![Fraud_Data class distribution](figures/fraud_class_distribution.png)

*Figure 1: Fraud_Data class distribution (~9.4% fraud).*

![Credit card class distribution](figures/creditcard_class_distribution.png)

*Figure 2: creditcard.csv class distribution (log scale; ~0.17% fraud).*

---

## 4. Data Cleaning and Preprocessing

Implemented in `src/preprocessing/`; entry point `python scripts/run_preprocess.py`.

**Fraud_Data:** snake_case columns → parse timestamps → validate required fields → coerce dtypes.

**creditcard.csv:** snake_case → remove **1,081** duplicates → numeric cast for `time`, `amount`, `class`, and PCA features → save `data/processed/creditcard_clean.csv`.

Both streams share the same cleaning utilities and unit tests.

---

## 5. Exploratory Data Analysis

### E-commerce (Fraud_Data)

| Class | Median signup-to-purchase time |
|-------|-------------------------------|
| Legitimate | ~1,443 hours (~60 days) |
| Fraud | ~0 hours |

| Segment | Fraud rate |
|---------|------------|
| Direct | 10.54% |
| Ads | 9.21% |
| SEO | 8.93% |

Median `purchase_value` (~35) and `age` (~33) are similar across classes.

![Signup-to-purchase time by class](figures/time_since_signup_by_class.png)

*Figure 3: Signup-to-purchase time by class (e-commerce).*

### Banking (creditcard.csv)

| Metric | Legitimate | Fraud |
|--------|------------|-------|
| Share | 99.83% | 0.17% |
| Median amount | €22.00 | €9.82 |
| Median time (s) | 84,711 | 73,408 |

EDA motivates **different feature stories** under one modeling framework: behavioral/geo for e-commerce; PCA + amount/time for banking.

---

## 6. Geolocation Enrichment

Applies to **Fraud_Data only** (`scripts/run_geolocation_enrichment.py`):

| Metric | Value |
|--------|-------|
| Matched to country | 129,146 (**85.46%**) |
| Unmatched (`unknown`) | 21,966 |
| Distinct countries | 182 |

Creditcard features are anonymized PCA components — no IP geolocation is possible or required.

---

## 7. Feature Engineering

### E-commerce

| Family | Features |
|--------|----------|
| Temporal | `time_since_signup_hours`, `hour_of_day`, `day_of_week` |
| Velocity | rolling txn counts (uniform in this data — one txn/user) |
| Encoding | scaled numerics + one-hot `source`/`browser`/`sex`/`country` |

**Output:** `fraud_data_features.csv` — 151,112 × 204 (203 features + label).

### Banking

| Family | Features |
|--------|----------|
| PCA | `v1`–`v28` (anonymized) |
| Scaled | `time`, `amount` via `StandardScaler` |

**Output:** loaded via `load_creditcard_feature_matrix()` — 283,726 × 30 features + label.

Shared API: `stratified_train_test_split`, SMOTE, tune → train → report → SHAP.

---

## 8. Class Imbalance Handling

| Stream | Train minority | SMOTE strategy | Holdout prevalence |
|--------|----------------|----------------|--------------------|
| Fraud_Data | 11,321 / 120,889 (9.36%) | Full balance to 50/50 | 9.36% (2,830 / 30,223) |
| creditcard | 378 / 226,980 (0.17%) | Partial (`sampling_strategy=0.05`) | 0.17% (95 / 56,746) |

**Safeguards (both streams):** stratified 80/20 split (`RANDOM_STATE=42`); SMOTE on train only; test never resampled. Creditcard uses partial oversampling to avoid exploding a 226k majority class into a half-million-row train set while still lifting minority exposure.

![SMOTE class distribution](figures/smote_class_distribution.png)

*Figure 4: Fraud_Data train class balance before/after SMOTE.*

---

## 9. Modeling Approach

**Unified workflow** (`src/modeling/workflow.py`):

| Step | Fraud_Data | creditcard |
|------|------------|------------|
| Loader | `load_fraud_feature_matrix` | `load_creditcard_feature_matrix` |
| Models | LR (GridSearchCV), RF & XGBoost (RandomizedSearchCV) | Same three models |
| CV metric | PR-AUC | PR-AUC |
| Selection | Holdout PR-AUC | Holdout PR-AUC |
| Outputs | `reports/outputs/` | `reports/outputs/creditcard/` |
| CLI | `run_modeling_reports.py` | `run_creditcard_modeling.py` |
| Unified | `run_unified_modeling.py` | |

Tuning uses a stratified subsample (50k rows) for creditcard hyperparameter search; final models refit on the full SMOTE-resampled training set.

---

## 10. Model Comparison

### E-commerce (Fraud_Data) — holdout 30,223

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|-----|---------|--------|
| **random_forest_tuned** | **0.9914** | **0.5272** | **0.6884** | **0.7696** | **0.6246** |
| logistic_regression | 0.1691 | 0.6912 | 0.2718 | 0.7423 | 0.3864 |

Confusion (best): TP **1,492** · FP **13** · FN **1,338** · TN **27,380**

![E-commerce PR curve](../reports/outputs/plots/model_comparison_pr_curve.png)

*Figure 5: Fraud_Data precision-recall comparison.*

![E-commerce confusion matrix](../reports/outputs/confusion_matrices/confusion_matrix_random_forest_tuned.png)

*Figure 6: Fraud_Data best-model confusion matrix.*

### Banking (creditcard) — holdout 56,746

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|-----|---------|--------|
| **random_forest** | **0.9231** | **0.7579** | **0.8324** | **0.9750** | **0.8094** |
| xgboost | 0.8506 | 0.7789 | 0.8132 | 0.9776 | 0.8094 |
| logistic_regression | 0.6757 | 0.7895 | 0.7282 | 0.9671 | 0.6937 |

Confusion (best): TP **72** · FP **6** · FN **23** · TN **56,645**

![Creditcard PR curve](../reports/outputs/creditcard/plots/model_comparison_pr_curve.png)

*Figure 7: creditcard precision-recall comparison.*

![Creditcard ROC curve](../reports/outputs/creditcard/plots/model_comparison_roc_curve.png)

*Figure 8: creditcard ROC comparison.*

![Creditcard confusion matrix](../reports/outputs/creditcard/confusion_matrices/confusion_matrix_random_forest.png)

*Figure 9: creditcard best-model confusion matrix.*

![Creditcard feature importance](../reports/outputs/creditcard/plots/best_model_feature_importance.png)

*Figure 10: creditcard top feature importance (PCA components).*

**Cross-stream takeaway:** Random Forest wins PR-AUC on both streams. E-commerce favors extreme precision; banking achieves higher recall at still-strong precision because PCA features separate fraud more cleanly despite rarer positives.

---

## 11. SHAP Explainability

Both streams produce summary, bar, waterfall, and dependence plots.

### E-commerce

Top SHAP drivers align with feature importance: `time_since_signup_hours`, geography, browser, channel, timing.

![SHAP summary — Fraud_Data](../reports/figures/shap_summary_plot.png)

*Figure 11: Fraud_Data SHAP beeswarm.*

![SHAP bar — Fraud_Data](../reports/figures/shap_bar_plot.png)

*Figure 12: Fraud_Data mean |SHAP|.*

![SHAP waterfall — Fraud_Data](../reports/figures/shap_waterfall_plot.png)

*Figure 13: Fraud_Data single-case waterfall.*

![SHAP dependence — Fraud_Data](../reports/figures/shap_dependence_plot.png)

*Figure 14: Fraud_Data dependence plot.*

### Banking

Top model importance: **V14 (22.8%)**, **V17 (15.4%)**, **V10 (13.3%)**, **V12 (11.8%)**, **V16 (7.6%)**. SHAP artifacts under `reports/figures/creditcard/` and `reports/shap/creditcard/`.

> **PCA note:** `v1`–`v28` are anonymized. SHAP ranks contribution to scores but cannot recover merchant-facing attributes. Use for model monitoring and analyst prioritization, not customer-facing decline rules labeled by “country” or “channel.”

![SHAP summary — creditcard](../reports/figures/creditcard/shap_summary_plot.png)

*Figure 15: creditcard SHAP beeswarm.*

![SHAP bar — creditcard](../reports/figures/creditcard/shap_bar_plot.png)

*Figure 16: creditcard mean |SHAP|.*

![SHAP waterfall — creditcard](../reports/figures/creditcard/shap_waterfall_plot.png)

*Figure 17: creditcard single-case waterfall.*

![SHAP dependence — creditcard](../reports/figures/creditcard/shap_dependence_plot.png)

*Figure 18: creditcard SHAP dependence (top PCA feature).*

CLI: `python scripts/run_shap_analysis.py` · `python scripts/run_creditcard_shap.py` (or `run_creditcard_shap_fast.py`)

---

## 12. Key Fraud Insights

1. **One framework, two feature worlds** — Shared split/SMOTE/tune/evaluate/SHAP stack; e-commerce uses behavior + geo; banking uses PCA + amount/time.
2. **Signup velocity dominates e-commerce** — ~69% of RF importance; actionable via new-account cooldowns without the model alone.
3. **PCA components dominate banking** — V14/V17/V10 drive scores; amount ranks lower — threshold rules on amount alone are insufficient.
4. **Both streams support review-first ops** — 13 FP (e-commerce) and 6 FP (banking) on holdouts keep analyst noise low.
5. **Prevalence drives metric choice** — Accuracy is misleading especially on creditcard (99.8% majority); **PR-AUC** is the selection metric for both.
6. **Responsible AI** — Geography/channel triage e-commerce queues; banking explanations stay model-internal due to anonymization.

---

## 13. Recommendations

### Unified operating model

- Deploy **one case-management workflow** with stream tag (`ecommerce` | `card`).
- Score with the stream-specific best model; shared SLA for review turnaround.
- Primary KPI: precision and alert volume; secondary: recall / missed fraud cost.

### E-commerce-specific

- Cooldown / step-up verification when `time_since_signup_hours` is below 24–72 hours.
- Monitor Direct channel (10.54% fraud rate); route `country_unknown` to verification.

### Banking-specific

- Prefer model score + amount context over amount-only rules.
- Monitor drift on top PCA components (V14, V17, V10); retrain when SHAP rankings shift.
- Do not invent business labels for PCA features in customer communications.

### Operational monitoring

- Weekly false-negative review on both streams.
- Quarterly re-tune + SHAP refresh (`run_unified_modeling.py` + SHAP scripts).

---

## 14. Limitations

1. **E-commerce velocity features** — One transaction per user; repeat-purchase fraud not observable.
2. **IP coverage** — 14.5% unknown countries.
3. **E-commerce recall** — ~47% of holdout fraud missed at default threshold.
4. **Banking anonymization** — PCA features limit business-facing explanations.
5. **Partial SMOTE on creditcard** — `sampling_strategy=0.05` is a pragmatic trade-off; class weights / ADASYN not fully benchmarked.
6. **Threshold not business-calibrated** — Default 0.5 may not match chargeback cost ratios.

---

## 15. Future Work

1. Threshold optimization on PR curves per stream (cost-sensitive).
2. Compare SMOTE vs class weights vs ADASYN on both datasets.
3. Ensemble / stacking across streams for portfolio-level risk.
4. Persist trained model binaries for production scoring APIs.
5. Live monitoring: feature drift + precision/recall dashboards.
6. Revisit e-commerce velocity when multi-txn user histories are available.

---

## 16. Conclusion

This challenge delivers a **unified fraud detection solution across e-commerce and banking transaction streams**. The same reproducible stack — clean → enrich/engineer → imbalance-aware train → tune on PR-AUC → holdout evaluate → SHAP — is applied to both `Fraud_Data.csv` and `creditcard.csv`.

| Stream | Best model | PR-AUC | Precision | Recall |
|--------|------------|--------|-----------|--------|
| E-commerce | `random_forest_tuned` | 0.625 | 99.1% | 52.7% |
| Banking | `random_forest` | 0.809 | 92.3% | 75.8% |

E-commerce is ready for **high-precision review queues** plus signup-timing policy controls. Banking achieves stronger ranking performance under extreme imbalance, with PCA-driven explainability for analysts and model ops. Together they close the original knowledge gap: a complete, dual-stream fraud analytics system suitable for submission and stakeholder review.

---

## Appendix: Reproducibility

```bash
pip install -r requirements.txt
pytest tests/ -v

# Shared preparation
python scripts/run_preprocess.py
python scripts/run_geolocation_enrichment.py
python scripts/run_feature_engineering.py
python scripts/run_imbalance_resampling.py

# Unified modeling (both streams)
python scripts/run_unified_modeling.py
# or separately:
python scripts/run_modeling_reports.py
python scripts/run_creditcard_modeling.py

# Explainability
python scripts/run_shap_analysis.py
python scripts/run_creditcard_shap.py
python scripts/run_fraud_insights.py

# Reports & dashboard
python scripts/build_final_report_html.py
python scripts/generate_final_report.py
streamlit run dashboard/app.py
```

**Configuration:** `RANDOM_STATE=42`, `TEST_SIZE=0.2` in `src/config.py`

| Artifact | Location |
|----------|----------|
| E-commerce metrics | `reports/outputs/` or `reports/modeling/` |
| Banking metrics | `reports/outputs/creditcard/` |
| E-commerce SHAP | `reports/figures/`, `reports/shap/` |
| Banking SHAP | `reports/figures/creditcard/`, `reports/shap/creditcard/` |
| Final report (MD) | `docs/final-report.md` |
| Final report (HTML) | `docs/final-report.html` / `reports/final_report.html` |

*Export to PDF: open the HTML report in a browser → Print → Save as PDF.*

# Executive Fraud Detection Insights — Fraud_Data

## Executive summary

The tuned **random_forest_tuned** model is the current best performer on the e-commerce fraud dataset (PR-AUC **0.625**). It flags fraud with **99.1% precision** and **52.7% recall** on the holdout set — meaning most alerts are real fraud, but roughly half of fraudulent transactions still evade detection at the default threshold.

On **30,223** holdout transactions, the model caught **1,492** fraud cases, missed **1,338**, and generated only **13** false alarms. The dominant risk signal is **short time between signup and purchase**, which accounts for the majority of the model's feature-importance weight.

---

## Top fraud indicators

Ranked by the best model's feature importance (Random Forest):

| Rank | Feature | Importance | What it signals |
| --- | --- | --- | --- |
| 1 | `time_since_signup_hours` | 0.6887 | short time between signup and purchase |
| 2 | `country_United States` | 0.0460 | transactions from United States |
| 3 | `day_of_week` | 0.0377 | purchase day of week |
| 4 | `hour_of_day` | 0.0266 | purchase hour of day |
| 5 | `age` | 0.0220 | customer age |
| 6 | `purchase_value` | 0.0189 | higher purchase amounts |
| 7 | `country_China` | 0.0149 | transactions from China |
| 8 | `country_unknown` | 0.0148 | transactions from unknown |
| 9 | `browser_Chrome` | 0.0116 | using Chrome browser |
| 10 | `country_Germany` | 0.0115 | transactions from Germany |

**Key takeaway:** `time_since_signup_hours` dominates (~69% of importance), consistent with EDA showing fraudsters purchase almost immediately after signup while legitimate customers wait weeks on average.

---

## High-risk customer behaviors

Patterns the model and EDA associate with elevated fraud risk:

1. **Immediate post-signup purchases** — Median signup-to-purchase time is near **0 hours** for fraud vs **~1,443 hours** for legitimate users. First-purchase velocity is the strongest behavioral red flag.
2. **New-account activity without relationship history** — Fraud cases cluster around accounts that transact before normal trust patterns develop.
3. **Channel-specific behavior** — Direct traffic showed the highest fraud rate (~10.5%) in EDA; channel features contribute secondary signal.
4. **Geographic context** — Country labels (e.g. United States, China, unknown IP mapping) add secondary risk context; **4** geographic features rank in the top 10.

---

## High-risk transaction characteristics

Transaction-level patterns linked to higher model scores:

- **Timing:** Purchases at certain hours/days of week contribute measurable risk (`hour_of_day`, `day_of_week` in top drivers).
- **Value:** `purchase_value` ranks in top features but EDA shows similar medians across classes — amount alone is a weak rule, but adds context in combination with other signals.
- **Device/browser:** Browser one-hot features (Chrome, IE, Safari, etc.) appear in the top 20, suggesting device fingerprint patterns differ between fraud and legitimate traffic.
- **Geolocation:** Transactions from specific countries or with unmapped IPs (`country_unknown`) can elevate scores — treat as review signals, not automatic blocks.

---

## Recommendations

### Fraud prevention

1. **Cooldown on new accounts** — Apply stepped purchase limits or extra verification when `time_since_signup_hours` is below a business-defined threshold (e.g. 24–72 hours).
2. **First-transaction rules** — Combine signup timing with purchase value and channel for a lightweight rules layer ahead of the ML model.
3. **Channel monitoring** — Review Direct-traffic conversion funnels; higher fraud rates may indicate acquisition fraud or weak signup controls on that path.
4. **Do not block by country alone** — Geographic features inform prioritization, not blanket denial.

### Customer verification

1. **Step-up verification for high-score new users** — Email, phone, or payment 3-D Secure when signup-to-purchase time is very short and the model score exceeds threshold.
2. **Manual review queue** — With **99%+ precision**, model-flagged cases are strong review candidates; analysts can focus on the highest-scored transactions first.
3. **Unknown geography handling** — Transactions with unmapped IP/country should route to verification rather than auto-approval.

### Operational monitoring

1. **Track recall, not just precision** — Current recall (~53%) means ~47% of fraud is missed; monitor `false_negatives` weekly and tune thresholds if missed fraud is too costly.
2. **Alert volume dashboard** — With only **13** false positives on holdout, the model is operationally efficient; watch for drift if alert volume spikes.
3. **Feature drift alerts** — Monitor distributions of `time_since_signup_hours`, top countries, and channel mix; fraud tactics shift over time.
4. **Retrain cadence** — Re-run tuning and SHAP analysis quarterly or after major fraud incidents.

---

## For data scientists

- **Best model:** `random_forest_tuned` selected by PR-AUC (0.6246) vs logistic regression (0.3864).
- **Metrics:** precision=0.9914, recall=0.5272, F1=0.6884, ROC-AUC=0.7696.
- **Imbalance:** SMOTE on train only; holdout retains ~9.4% fraud prevalence.
- **Feature dominance:** A single engineered feature (`time_since_signup_hours`) drives most importance — validate with SHAP and test threshold policies for recall improvement.
- **Velocity features:** Uniform in current data (one transaction per user); exclude or deprioritize until repeat-purchase data is available.
- **Next steps:** Threshold tuning on PR curve, SHAP case review, and stream-specific ops playbooks for both Fraud_Data and creditcard.

## Unified solution (both streams)

- **E-commerce (`Fraud_Data`):** best model `random_forest_tuned` — high precision (~99%), dominant signal `time_since_signup_hours`.
- **Banking (`creditcard.csv`):** best model `random_forest` — PR-AUC ~0.81, recall ~76%, dominant PCA drivers `V14` / `V17` / `V10` (see `reports/outputs/creditcard/`).
- **Shared stack:** stratified split → SMOTE on train → tune LR/RF/XGBoost on PR-AUC → holdout evaluate → SHAP (`run_unified_modeling.py`).

## For fraud analysts

- **Prioritize:** New accounts buying within hours of signup — this is the clearest pattern.
- **Review queue:** Model alerts are highly precise; investigate flagged orders before release.
- **Context matters:** Combine model score with payment history, customer contact verification, and shipping/billing mismatches.
- **Geography:** Elevated country signals mean "review harder," not "decline automatically."
- **Volume expectation:** On a similar holdout, expect ~13 false alarms per 30,223 transactions at the current threshold — very low analyst noise.

## For executives

- **Business problem:** ~9.4% of e-commerce transactions are fraudulent; undetected fraud has direct revenue and chargeback cost.
- **Model readiness:** Strong precision (99%+) supports a **review-first** deployment — the model reliably tells you which cases to investigate, not which to auto-block without human oversight.
- **Gap to close:** Recall near 53% means roughly half of fraud still slips through at default settings; investment in threshold tuning and verification workflows will improve capture rate.
- **Highest-ROI control:** Policies around **new-account purchase timing** address the single strongest signal and do not require model inference alone.
- **Responsible AI:** Use geographic and channel signals for triage, not discriminatory blocking; document decisions and audit flagged cases regularly.

---

## Source artifacts

- Model metrics: `reports\modeling\model_comparison_metrics.csv`
- Best model summary: `reports\modeling\best_model_summary.csv`
- Feature importance: `reports\modeling\top_features_best_model.csv`
- EDA reference: `notebooks/eda-fraud-data.ipynb`
- Class imbalance: `reports/class_imbalance_summary.md`
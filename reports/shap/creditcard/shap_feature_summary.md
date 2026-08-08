# SHAP Explainability Summary — creditcard

## Model explained
- **Model:** `random_forest`
- **Holdout PR-AUC:** 0.8094
- **Holdout F1:** 0.8324

## How to read these results

SHAP shows **which features pushed a transaction toward or away from a fraud alert**. 
It does not prove causation, but it helps analysts understand why the model scored 
a case highly and which patterns deserve operational attention.

> **PCA note:** `v1`–`v28` are anonymized principal components. SHAP ranks their 
> contribution to model scores but cannot recover original merchant-facing attributes.

## Top 10 fraud predictors

### 1. `v14`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V14 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 2. `v12`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V12 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 3. `v4`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V4 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 4. `v10`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V10 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 5. `v17`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V17 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 6. `v3`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V3 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 7. `v11`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V11 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 8. `v16`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V16 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 9. `v2`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V2 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 10. `v7`

**Why it matters:** PCA components capture anonymized correlations in the original card transaction features. They are not directly interpretable as merchant-facing business rules.

**Impact on fraud probability:** Higher values for **PCA component V7 (anonymized credit-card feature)** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

## Example case (waterfall plot)

The waterfall chart uses explained holdout row **54513** to show how individual 
features raised or lowered the fraud score for one transaction. This is useful when 
an analyst asks, "Why was this specific order flagged?"

## Responsible use

- PCA features should guide **model monitoring and retraining**, not customer-facing decline rules.
- Combine SHAP rankings with amount/time thresholds and investigator review for card streams.
- SHAP reflects historical patterns; new fraud tactics may not appear until retraining.
- Combine model explanations with policy, customer context, and investigator judgment.
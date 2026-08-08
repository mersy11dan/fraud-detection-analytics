# SHAP Explainability Summary — Fraud_Data

## Model explained
- **Model:** `random_forest`
- **Holdout PR-AUC:** 0.6243
- **Holdout F1:** 0.6887

## How to read these results

SHAP shows **which features pushed a transaction toward or away from a fraud alert**. 
It does not prove causation, but it helps analysts understand why the model scored 
a case highly and which patterns deserve operational attention.

## Top 10 fraud predictors

### 1. `time_since_signup_hours`

**Why it matters:** Fraudsters often buy soon after creating an account, before normal trust patterns develop.

**Impact on fraud probability:** Higher values for **short time between signup and purchase** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 2. `country_United States`

**Why it matters:** Some regions appear more often in flagged transactions in this dataset. This is a review signal, not proof that a country is inherently risky.

**Impact on fraud probability:** Higher values for **transactions from United States** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 3. `browser_Chrome`

**Why it matters:** Device and browser choices sometimes differ between automated fraud and normal shopping behavior.

**Impact on fraud probability:** Higher values for **using Chrome browser** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 4. `day_of_week`

**Why it matters:** Purchase timing can differ when fraud activity clusters at unusual hours or days.

**Impact on fraud probability:** Higher values for **purchase day of week** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 5. `source_Direct`

**Why it matters:** Acquisition channel can reveal how a customer arrived and whether that path matches typical fraud patterns.

**Impact on fraud probability:** Higher values for **traffic from Direct channel** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 6. `hour_of_day`

**Why it matters:** Purchase timing can differ when fraud activity clusters at unusual hours or days.

**Impact on fraud probability:** Higher values for **purchase hour of day** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 7. `country_Germany`

**Why it matters:** Some regions appear more often in flagged transactions in this dataset. This is a review signal, not proof that a country is inherently risky.

**Impact on fraud probability:** Higher values for **transactions from Germany** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 8. `country_unknown`

**Why it matters:** Some regions appear more often in flagged transactions in this dataset. This is a review signal, not proof that a country is inherently risky.

**Impact on fraud probability:** Higher values for **transactions from unknown** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 9. `age`

**Why it matters:** Customer age may correlate with different purchasing and risk patterns in the training data.

**Impact on fraud probability:** Higher values for **customer age** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

### 10. `purchase_value`

**Why it matters:** Unusually large purchases can signal stolen payment methods or test transactions.

**Impact on fraud probability:** Higher values for **higher purchase amounts** tend to **decrease** the model's fraud score, making a fraud flag less likely.

**Business interpretation:** When this signal is strong, fraud teams should treat it as a 
**prioritization clue** for manual review or rule design—not as an automatic block on its own.

## Example case (waterfall plot)

The waterfall chart uses explained holdout row **23286** to show how individual 
features raised or lowered the fraud score for one transaction. This is useful when 
an analyst asks, "Why was this specific order flagged?"

## Responsible use

- Geographic and channel signals should inform **review queues**, not blanket customer rejection.
- SHAP reflects patterns in historical data; new fraud tactics may not appear until retraining.
- Combine model explanations with policy, customer context, and investigator judgment.
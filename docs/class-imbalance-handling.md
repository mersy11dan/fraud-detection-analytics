# Class Imbalance Handling

## Problem

Fraud_Data has a minority fraud class of roughly **9–10%**, with an imbalance ratio near **9.7:1**. Models trained on imbalanced data often predict the majority class too often, which inflates accuracy but hurts fraud recall.

## Technique choice: SMOTE on the training set

We apply **SMOTE (Synthetic Minority Over-sampling Technique)** to the **training split only**.

| Approach | Why we chose / rejected it |
|----------|----------------------------|
| **SMOTE (selected)** | Keeps all legitimate transactions in the training set and synthesizes minority examples in feature space. Better for fraud use cases where discarding non-fraud rows wastes information. |
| **Random undersampling (rejected as default)** | Would throw away most legitimate transactions (~90% of rows) to balance classes. Simple, but wasteful for this dataset size and business context. |
| **Resampling the test set (rejected)** | Never applied. The test set must reflect the real production class distribution so precision/recall metrics are meaningful. |

## Pipeline safeguards

1. `train_test_split(..., stratify=y)` creates a holdout test set first.
2. Resampling runs on `(X_train, y_train)` only.
3. `(X_test, y_test)` is returned unchanged.
4. Before/after class distributions are logged and saved to `reports/` for the interim submission.

## Usage

```bash
python scripts/run_imbalance_resampling.py
```

```python
from src.modeling.imbalance import prepare_resampled_training_data

result = prepare_resampled_training_data(features, target, strategy="smote")
X_train, y_train = result.x_train_resampled, result.y_train_resampled
X_test, y_test = result.x_test, result.y_test  # never resampled
```

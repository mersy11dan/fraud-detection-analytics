# Best Fraud Model Summary — Fraud_Data

## Selected model
- **Model:** `random_forest_tuned`
- **Selection metric:** `auc_pr` (higher is better for imbalanced fraud detection)
- **Holdout AUC-PR:** 0.6246
- **Holdout F1:** 0.6884
- **Holdout recall:** 0.5272
- **Holdout precision:** 0.9914

## Comparison vs runner-up
- **Runner-up:** `logistic_regression`
- **AUC-PR gap:** +0.2382
- **F1 gap:** +0.4166

## Model comparison table

| model_name | accuracy | precision | recall | f1 | roc_auc | auc_pr |
| --- | --- | --- | --- | --- | --- | --- |
| random_forest_tuned | 0.9553 | 0.9914 | 0.5272 | 0.6884 | 0.7696 | 0.6246 |
| logistic_regression | 0.6532 | 0.1691 | 0.6912 | 0.2718 | 0.7423 | 0.3864 |

## Best model confusion matrix

- True positives (fraud caught): **1,492**
- False negatives (missed fraud): **1,338**
- False positives (false alarms): **13**
- True negatives (correctly cleared): **27,380**

## Top feature drivers

| feature | importance |
| --- | --- |
| time_since_signup_hours | 0.6887 |
| country_United States | 0.0460 |
| day_of_week | 0.0377 |
| hour_of_day | 0.0266 |
| age | 0.0220 |

### Plain-language notes
- `time_since_signup_hours`: short time between signup and purchase
- `country_United States`: transactions from United States
- `day_of_week`: purchase day of week
- `hour_of_day`: purchase hour of day
- `age`: customer age

## Interim recommendation

`random_forest_tuned` is the current best model on the stratified holdout when ranked by **AUC-PR**. It should be carried forward for threshold tuning and deeper explainability work (SHAP), but is not yet production-ready without those follow-up steps.

## Report artifacts

Reuse the companion files in this folder:

- `model_comparison_metrics.csv` — full metrics table for all models
- `best_model_summary.csv` — one-row summary for the best model
- `top_features_best_model.csv` — ranked features for the best model
- `plots/model_comparison_pr_curve.png` — precision-recall comparison chart
- `plots/best_model_confusion_matrix.png` — best-model confusion matrix
- `plots/best_model_feature_importance.png` — top-10 feature importance chart
# Best Fraud Model Summary — creditcard

## Selected model
- **Model:** `random_forest`
- **Selection metric:** `pr_auc` (higher is better for imbalanced fraud detection)
- **Holdout PR-AUC:** 0.8094
- **Holdout F1:** 0.8324
- **Holdout recall:** 0.7579
- **Holdout precision:** 0.9231
- **Holdout ROC-AUC:** 0.9750

## Comparison vs runner-up
- **Runner-up:** `xgboost`
- **PR-AUC gap:** +0.0001
- **F1 gap:** +0.0192

## Model comparison table

| model_name | precision | recall | f1 | roc_auc | pr_auc | true_positives | false_positives | false_negatives | true_negatives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| random_forest | 0.9231 | 0.7579 | 0.8324 | 0.9750 | 0.8094 | 72 | 6 | 23 | 56645 |
| xgboost | 0.8506 | 0.7789 | 0.8132 | 0.9776 | 0.8094 | 74 | 13 | 21 | 56638 |
| logistic_regression | 0.6757 | 0.7895 | 0.7282 | 0.9671 | 0.6937 | 75 | 36 | 20 | 56615 |

## Best model confusion matrix

- True positives (fraud caught): **72**
- False negatives (missed fraud): **23**
- False positives (false alarms): **6**
- True negatives (correctly cleared): **56,645**

## Top feature drivers

| feature | importance |
| --- | --- |
| v14 | 0.2277 |
| v17 | 0.1540 |
| v10 | 0.1331 |
| v12 | 0.1179 |
| v16 | 0.0761 |

### Plain-language notes
- `v14`: PCA component V14 (anonymized credit-card feature)
- `v17`: PCA component V17 (anonymized credit-card feature)
- `v10`: PCA component V10 (anonymized credit-card feature)
- `v12`: PCA component V12 (anonymized credit-card feature)
- `v16`: PCA component V16 (anonymized credit-card feature)

## Interim recommendation

`random_forest` is the current best model on the stratified holdout when ranked by **PR-AUC**. It should be carried forward for threshold tuning and deeper explainability work (SHAP), but is not yet production-ready without those follow-up steps.

## Report artifacts

- `model_comparison_metrics.csv` — full metrics table for all models
- `best_model_summary.csv` — one-row summary for the best model
- `top_features_best_model.csv` — ranked features for the best model
- `plots/model_comparison_pr_curve.png` — precision-recall comparison chart
- `plots/model_comparison_roc_curve.png` — ROC comparison chart
- `confusion_matrices/` — per-model confusion matrix plots
- `plots/best_model_feature_importance.png` — top-10 feature importance chart
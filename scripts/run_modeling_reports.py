"""CLI entry point for report-ready fraud modeling outputs."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.config import RANDOM_STATE
from src.modeling.data import load_fraud_feature_matrix, stratified_train_test_split
from src.modeling.reporting import save_modeling_report
from src.modeling.training import train_classifier
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [12, 20, None],
    "min_samples_leaf": [1, 5],
}


def main() -> None:
    """Train comparison models and write report-ready artifacts to reports/modeling/."""
    features, target = load_fraud_feature_matrix(logger=LOGGER)
    split = stratified_train_test_split(features, target, random_state=RANDOM_STATE)

    smote = SMOTE(random_state=RANDOM_STATE)
    x_train_resampled, y_train_resampled = smote.fit_resample(split.x_train, split.y_train)
    x_train_resampled = pd.DataFrame(x_train_resampled, columns=split.feature_names)
    y_train_resampled = pd.Series(y_train_resampled, name="class")

    baseline_result = train_classifier(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        x_train_resampled,
        y_train_resampled,
        split.x_test,
        split.y_test,
        model_name="logistic_regression",
        logger=LOGGER,
    )

    LOGGER.info("Tuning Random Forest on pre-SMOTE training split")
    rf_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=RF_PARAM_GRID,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
    )
    rf_search.fit(split.x_train, split.y_train)
    LOGGER.info("Best RF params: %s", rf_search.best_params_)

    ensemble_result = train_classifier(
        RandomForestClassifier(
            **rf_search.best_params_,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        x_train_resampled,
        y_train_resampled,
        split.x_test,
        split.y_test,
        model_name="random_forest_tuned",
        logger=LOGGER,
    )

    report_paths = save_modeling_report(
        [baseline_result, ensemble_result],
        logger=LOGGER,
    )

    print("\nReport-ready modeling outputs:")
    for name, path in report_paths.as_dict().items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

"""CLI entry point for baseline fraud classifier training."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.data import load_fraud_feature_matrix
from src.modeling.imbalance import prepare_resampled_training_data
from src.modeling.training import (
    compare_model_results,
    save_model_metrics,
    train_multiple_classifiers,
)
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)
DEFAULT_MODELS = ["logistic_regression", "random_forest", "xgboost"]


def main() -> None:
    """Train baseline classifiers on processed Fraud_Data and save metrics."""
    features, target = load_fraud_feature_matrix(logger=LOGGER)
    resampled = prepare_resampled_training_data(
        features,
        target,
        strategy="smote",
        logger=LOGGER,
    )

    results = train_multiple_classifiers(
        DEFAULT_MODELS,
        resampled.x_train_resampled,
        resampled.y_train_resampled,
        resampled.x_test,
        resampled.y_test,
        store_training_data=False,
        logger=LOGGER,
    )

    comparison = compare_model_results(results)
    output_path = save_model_metrics(comparison, logger=LOGGER)

    print(comparison.to_string(index=False))
    LOGGER.info("Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()

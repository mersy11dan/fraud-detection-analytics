"""CLI entry point for class imbalance handling on Fraud_Data features."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_DIR, FRAUD_DATA_FEATURES_FILENAME
from src.features.pipeline import build_fraud_feature_matrix
from src.modeling.imbalance import prepare_resampled_training_data, save_imbalance_report
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Build features, resample training data, and write interim imbalance report."""
    features_path = DATA_PROCESSED_DIR / FRAUD_DATA_FEATURES_FILENAME
    if features_path.exists():
        LOGGER.info("Loading saved feature matrix from %s", features_path)
        feature_table = __import__("pandas").read_csv(features_path)
        target = feature_table.pop("class")
        features = feature_table
    else:
        LOGGER.info("Feature matrix not found on disk; building from raw data")
        features, target, _ = build_fraud_feature_matrix(logger=LOGGER)

    result = prepare_resampled_training_data(
        features,
        target,
        strategy="smote",
        logger=LOGGER,
    )
    paths = save_imbalance_report(result, logger=LOGGER)

    print(result.summary_markdown)
    LOGGER.info("Report files: %s", paths)


if __name__ == "__main__":
    main()

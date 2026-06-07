"""CLI entry point for preprocessing all fraud detection datasets."""

from src.config import DATA_PROCESSED_DIR
from src.preprocessing.datasets import (
    preprocess_creditcard_data,
    preprocess_fraud_data,
    preprocess_ip_country_data,
)
from src.preprocessing.inspect import class_imbalance_summary, missing_values_summary
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

LOGGER = get_logger(__name__)


def main() -> None:
    """Load, preprocess, and save cleaned datasets to data/processed/."""
    output_dir = ensure_dir(DATA_PROCESSED_DIR)

    fraud_df = preprocess_fraud_data(logger=LOGGER)
    creditcard_df = preprocess_creditcard_data(logger=LOGGER)
    ip_country_df = preprocess_ip_country_data(logger=LOGGER)

    fraud_df.to_csv(output_dir / "fraud_data_clean.csv", index=False)
    creditcard_df.to_csv(output_dir / "creditcard_clean.csv", index=False)
    ip_country_df.to_csv(output_dir / "ip_country_clean.csv", index=False)

    LOGGER.info("Saved processed datasets to %s", output_dir)
    LOGGER.info("Fraud class distribution:\n%s", class_imbalance_summary(fraud_df, "class"))
    LOGGER.info(
        "Creditcard class distribution:\n%s",
        class_imbalance_summary(creditcard_df, "class"),
    )
    LOGGER.info("Fraud missing values after save:\n%s", missing_values_summary(fraud_df))


if __name__ == "__main__":
    main()

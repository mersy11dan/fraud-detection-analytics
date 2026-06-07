"""CLI entry point for Fraud_Data feature engineering."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pipeline import save_engineered_fraud_data
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Engineer, encode, scale, and save fraud detection features."""
    paths = save_engineered_fraud_data(logger=LOGGER)
    LOGGER.info("Engineered dataset: %s", paths["engineered"])
    LOGGER.info("Model-ready features: %s", paths["features"])


if __name__ == "__main__":
    main()

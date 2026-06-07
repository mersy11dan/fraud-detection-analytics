"""CLI entry point for fraud data geolocation enrichment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.geolocation import run_geolocation_enrichment_pipeline
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Enrich Fraud_Data with country labels and save analysis outputs."""
    enriched, patterns, output_path = run_geolocation_enrichment_pipeline(logger=LOGGER)

    LOGGER.info("Enriched dataset shape: %s", enriched.shape)
    LOGGER.info("Saved enriched data to: %s", output_path)
    LOGGER.info("Top countries by fraud rate (min 100 transactions):\n%s", patterns.head(15))


if __name__ == "__main__":
    main()

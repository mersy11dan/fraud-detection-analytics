"""CLI entry point for executive fraud insights."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.insights import save_fraud_insights
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Generate reports/fraud_insights.md from latest model outputs."""
    output_path = save_fraud_insights()
    LOGGER.info("Saved fraud insights to %s", output_path)
    print(f"Fraud insights written to: {output_path}")


if __name__ == "__main__":
    main()

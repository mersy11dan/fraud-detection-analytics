"""CLI entry point for tuned fraud classifier training (delegates to full workflow)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.workflow import run_fraud_modeling_workflow
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Train tuned LR, RF, and XGBoost models and save outputs to reports/outputs/."""
    workflow = run_fraud_modeling_workflow(logger=LOGGER)
    print(workflow.comparison.to_string(index=False))
    LOGGER.info("Best model: %s", workflow.best_result.model_name)


if __name__ == "__main__":
    main()

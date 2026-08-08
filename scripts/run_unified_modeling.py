"""CLI entry point for unified modeling across both transaction streams."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.workflow import run_creditcard_modeling_workflow, run_fraud_modeling_workflow
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Train tuned models on Fraud_Data and creditcard.csv."""
    fraud_workflow = run_fraud_modeling_workflow(logger=LOGGER)
    print("\n=== Fraud_Data best model ===")
    print(
        f"{fraud_workflow.best_result.model_name} "
        f"(PR-AUC={fraud_workflow.best_result.metrics.auc_pr:.4f})"
    )

    creditcard_workflow = run_creditcard_modeling_workflow(logger=LOGGER)
    print("\n=== creditcard.csv best model ===")
    print(
        f"{creditcard_workflow.best_result.model_name} "
        f"(PR-AUC={creditcard_workflow.best_result.metrics.auc_pr:.4f})"
    )


if __name__ == "__main__":
    main()

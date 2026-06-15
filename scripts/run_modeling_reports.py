"""CLI entry point for the full fraud modeling workflow."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.workflow import run_fraud_modeling_workflow
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Train tuned LR, RF, and XGBoost models and write outputs to reports/outputs/."""
    workflow = run_fraud_modeling_workflow(logger=LOGGER)

    print("\nModel comparison (sorted by PR-AUC):")
    print(workflow.comparison.to_string(index=False))
    print(
        f"\nBest model: {workflow.best_result.model_name} "
        f"(PR-AUC={workflow.best_result.metrics.auc_pr:.4f})"
    )

    if workflow.report_paths is not None:
        print("\nReport-ready outputs:")
        for name, path in workflow.report_paths.as_dict().items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

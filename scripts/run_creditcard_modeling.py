"""CLI entry point for creditcard modeling workflow."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.workflow import run_creditcard_modeling_workflow
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Train tuned models on creditcard.csv and write outputs to reports/outputs/creditcard/."""
    # Full three-model run with creditcard-optimized SMOTE/tuning inside the workflow.
    workflow = run_creditcard_modeling_workflow(
        logger=LOGGER,
        store_training_data=True,
    )

    print("\nCreditcard model comparison (sorted by PR-AUC):")
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

"""CLI entry point for creditcard SHAP explainability."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.shap_explain import run_creditcard_shap_explainability_workflow
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    """Export SHAP plots and summary for the best creditcard model."""
    analysis = run_creditcard_shap_explainability_workflow(logger=LOGGER)

    print(f"\nBest creditcard model explained: {analysis.model_name}")
    print("\nTop 10 SHAP features:")
    print(analysis.top_features.to_string(index=False))

    print("\nSHAP artifacts:")
    for name, path in analysis.paths.as_dict().items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

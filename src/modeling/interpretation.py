"""Feature importance and coefficient interpretation for fraud models."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import REPORTS_DIR
from src.modeling.training import ModelTrainingResult
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir


def extract_feature_importance(result: ModelTrainingResult) -> pd.DataFrame:
    """
    Extract interpretable feature rankings from the fitted model.

    Logistic Regression models return signed coefficients; tree ensembles return
    built-in ``feature_importances_``.
    """
    estimator = result.estimator
    features = result.feature_names

    if isinstance(estimator, LogisticRegression) or hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_).ravel()
        frame = pd.DataFrame(
            {
                "feature": features,
                "coefficient": coefficients,
                "importance": np.abs(coefficients),
                "direction": np.where(
                    coefficients > 0,
                    "increases_fraud_risk",
                    "decreases_fraud_risk",
                ),
            }
        )
        frame["importance_type"] = "coefficient"
    elif hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
        frame = pd.DataFrame(
            {
                "feature": features,
                "importance": importances,
                "coefficient": np.nan,
                "direction": "n/a",
            }
        )
        frame["importance_type"] = "feature_importance"
    else:
        raise TypeError(
            f"Model type {type(estimator).__name__} does not expose coefficients "
            "or feature_importances_."
        )

    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def get_coefficient_extremes(
    importance_table: pd.DataFrame,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the strongest positive and negative coefficients for linear models."""
    if importance_table["importance_type"].iloc[0] != "coefficient":
        raise ValueError("Coefficient extremes are only available for linear models.")

    positive = (
        importance_table[importance_table["coefficient"] > 0]
        .sort_values("coefficient", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    negative = (
        importance_table[importance_table["coefficient"] < 0]
        .sort_values("coefficient", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )
    return positive, negative


def save_top_features(
    importance_table: pd.DataFrame,
    *,
    model_name: str,
    output_dir: Path = REPORTS_DIR,
    top_n: int = 20,
    logger: logging.Logger | None = None,
) -> Path:
    """Persist the top-ranked features to CSV."""
    log = logger or get_logger(__name__)
    report_dir = ensure_dir(output_dir)
    safe_name = model_name.replace(" ", "_").lower()
    output_path = report_dir / f"top_features_{safe_name}.csv"

    importance_table.head(top_n).to_csv(output_path, index=False)
    log.info("Saved top %s features to %s", top_n, output_path)
    return output_path


def plot_top_features(
    importance_table: pd.DataFrame,
    *,
    model_name: str,
    top_n: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot the top features as a horizontal bar chart."""
    top_features = importance_table.head(top_n).iloc[::-1]
    plot_ax = ax or plt.subplots(figsize=(10, 6))[1]

    if top_features["importance_type"].iloc[0] == "coefficient":
        colors = np.where(top_features["coefficient"] > 0, "#d95f02", "#1b9e77")
        plot_ax.barh(top_features["feature"], top_features["coefficient"], color=colors)
        plot_ax.axvline(0, color="gray", linewidth=0.8)
        plot_ax.set_xlabel("Coefficient (positive → higher fraud risk)")
        title_suffix = "Strongest Coefficients"
    else:
        plot_ax.barh(top_features["feature"], top_features["importance"], color="#7570b3")
        plot_ax.set_xlabel("Feature importance")
        title_suffix = "Top Feature Importances"

    plot_ax.set_title(f"{title_suffix} — {model_name}")
    plot_ax.set_ylabel("")
    plt.tight_layout()
    return plot_ax


def describe_feature(feature_name: str) -> str:
    """Return a short plain-language label for a feature name."""
    if feature_name.startswith("country_"):
        return f"transactions from {feature_name.removeprefix('country_').replace('_', ' ')}"
    if feature_name.startswith("browser_"):
        return f"using {feature_name.removeprefix('browser_')} browser"
    if feature_name.startswith("source_"):
        return f"traffic from {feature_name.removeprefix('source_')} channel"
    if feature_name.startswith("sex_"):
        return f"customer sex = {feature_name.removeprefix('sex_')}"

    labels = {
        "purchase_value": "higher purchase amounts",
        "age": "customer age",
        "time_since_signup_hours": "short time between signup and purchase",
        "hour_of_day": "purchase hour of day",
        "day_of_week": "purchase day of week",
        "txn_count_last_1h": "many transactions in the last hour",
        "txn_count_last_24h": "many transactions in the last 24 hours",
        "txn_count_last_168h": "many transactions in the last week",
        "hours_since_last_txn": "time since the user's previous transaction",
        "user_cumulative_txn_count": "user transaction history volume",
        "user_txn_velocity_per_day": "high daily transaction velocity",
    }
    return labels.get(feature_name, feature_name.replace("_", " "))

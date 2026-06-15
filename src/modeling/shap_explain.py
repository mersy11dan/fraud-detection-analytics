"""SHAP explainability workflow for fraud detection models."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE, SHAP_FIGURES_DIR, SHAP_REPORTS_DIR
from src.modeling.data import load_fraud_feature_matrix, stratified_train_test_split
from src.modeling.interpretation import describe_feature
from src.modeling.training import ModelTrainingResult
from src.modeling.workflow import ModelingWorkflowResult, run_fraud_modeling_workflow
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

SUMMARY_PLOT = "shap_summary_plot.png"
BAR_PLOT = "shap_bar_plot.png"
WATERFALL_PLOT = "shap_waterfall_plot.png"
DEPENDENCE_PLOT = "shap_dependence_plot.png"
TOP_FEATURES_CSV = "shap_top_features.csv"
SUMMARY_MD = "shap_feature_summary.md"


@dataclass(frozen=True)
class ShapAnalysisPaths:
    """Paths to SHAP explainability artifacts."""

    output_dir: Path
    figures_dir: Path
    summary_plot: Path
    bar_plot: Path
    waterfall_plot: Path
    dependence_plot: Path
    top_features_csv: Path
    summary_markdown: Path

    def as_dict(self) -> dict[str, Path]:
        return asdict(self)


@dataclass
class ShapAnalysisResult:
    """Container for SHAP values and exported explainability artifacts."""

    model_name: str
    top_features: pd.DataFrame
    shap_values: np.ndarray
    explained_features: pd.DataFrame
    paths: ShapAnalysisPaths
    waterfall_index: int


def _sample_indices(n_rows: int, n_samples: int, random_state: int) -> np.ndarray:
    if n_rows <= n_samples:
        return np.arange(n_rows)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_rows, size=n_samples, replace=False))


def _resolve_matrices(
    result: ModelTrainingResult,
    *,
    background_size: int,
    explain_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if result.x_train is None:
        raise ValueError(
            "Training features are required for SHAP analysis. "
            "Re-run modeling with store_training_data=True."
        )

    x_background = result.x_train.iloc[
        _sample_indices(len(result.x_train), background_size, random_state)
    ].reset_index(drop=True)

    if result.x_test is not None:
        explain_indices = _sample_indices(len(result.x_test), explain_size, random_state + 1)
        x_explain = result.x_test.iloc[explain_indices].reset_index(drop=True)
        return x_background, x_explain, explain_indices

    features, target = load_fraud_feature_matrix()
    split = stratified_train_test_split(features, target, random_state=random_state)
    explain_indices = _sample_indices(len(split.x_test), explain_size, random_state + 1)
    x_explain = split.x_test.iloc[explain_indices].reset_index(drop=True)
    return x_background, x_explain, explain_indices


def _pick_local_waterfall_index(
    result: ModelTrainingResult,
    explain_indices: np.ndarray,
) -> int:
    """Pick an explained holdout row with strong fraud signal when available."""
    y_true = np.asarray(result.y_true.iloc[explain_indices])
    y_score = np.asarray(result.y_score[explain_indices])
    fraud_mask = y_true == 1
    if fraud_mask.any():
        fraud_local = np.where(fraud_mask)[0]
        return int(fraud_local[np.argmax(y_score[fraud_mask])])
    return int(np.argmax(y_score))


def create_shap_explainer(
    result: ModelTrainingResult,
    x_background: pd.DataFrame,
):
    """Create a SHAP explainer suited to the fitted model type."""
    model = result.estimator
    if isinstance(model, RandomForestClassifier) or hasattr(model, "estimators_"):
        return shap.TreeExplainer(model)
    if isinstance(model, LogisticRegression) or hasattr(model, "coef_"):
        return shap.LinearExplainer(model, x_background)
    return shap.Explainer(model.predict_proba, x_background)


def compute_shap_values(
    explainer,
    x_explain: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values for the positive (fraud) class when applicable."""
    values = explainer.shap_values(x_explain)

    if isinstance(values, list):
        return np.asarray(values[1])

    if hasattr(values, "values"):
        array = np.asarray(values.values)
        if array.ndim == 3:
            return array[:, :, 1]
        return array

    array = np.asarray(values)
    if array.ndim == 3:
        return array[:, :, 1]
    return array


def rank_top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank features by mean absolute SHAP value."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    ranked = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "impact_direction": np.where(
                mean_shap > 0,
                "increases_fraud_risk",
                "decreases_fraud_risk",
            ),
        }
    )
    return ranked.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def _business_impact_text(feature: str, mean_shap: float) -> str:
    """Plain-language description of how a feature moves fraud probability."""
    label = describe_feature(feature)
    if mean_shap > 0:
        return (
            f"Higher values for **{label}** tend to **increase** the model's fraud score, "
            "so these transactions are more likely to be flagged for review."
        )
    if mean_shap < 0:
        return (
            f"Higher values for **{label}** tend to **decrease** the model's fraud score, "
            "making a fraud flag less likely."
        )
    return f"**{label}** has limited average impact on fraud scores in this sample."


def _business_why_text(feature: str) -> str:
    """Why a feature may matter for fraud detection."""
    if feature == "time_since_signup_hours":
        return (
            "Fraudsters often buy soon after creating an account, before normal trust "
            "patterns develop."
        )
    if feature == "purchase_value":
        return "Unusually large purchases can signal stolen payment methods or test transactions."
    if feature.startswith("country_"):
        return (
            "Some regions appear more often in flagged transactions in this dataset. "
            "This is a review signal, not proof that a country is inherently risky."
        )
    if feature.startswith("source_"):
        return (
            "Acquisition channel can reveal how a customer arrived and whether that path "
            "matches typical fraud patterns."
        )
    if feature.startswith("browser_"):
        return (
            "Device and browser choices sometimes differ between automated fraud and "
            "normal shopping behavior."
        )
    if feature in {"hour_of_day", "day_of_week"}:
        return "Purchase timing can differ when fraud activity clusters at unusual hours or days."
    if feature == "age":
        return "Customer age may correlate with different purchasing and risk patterns in the training data."
    return (
        "This feature helps the model separate fraud from legitimate transactions based on "
        "patterns learned from historical data."
    )


def build_shap_summary_markdown(
    result: ModelTrainingResult,
    top_features: pd.DataFrame,
    *,
    waterfall_row: int,
) -> str:
    """Build a stakeholder-friendly markdown summary of SHAP findings."""
    metrics = result.metrics
    lines = [
        "# SHAP Explainability Summary — Fraud_Data",
        "",
        "## Model explained",
        f"- **Model:** `{result.model_name}`",
        f"- **Holdout PR-AUC:** {metrics.auc_pr:.4f}",
        f"- **Holdout F1:** {metrics.f1:.4f}",
        "",
        "## How to read these results",
        "",
        "SHAP shows **which features pushed a transaction toward or away from a fraud alert**. ",
        "It does not prove causation, but it helps analysts understand why the model scored ",
        "a case highly and which patterns deserve operational attention.",
        "",
        "## Top 10 fraud predictors",
        "",
    ]

    for rank, row in top_features.iterrows():
        feature = row["feature"]
        lines.extend(
            [
                f"### {rank + 1}. `{feature}`",
                "",
                f"**Why it matters:** {_business_why_text(feature)}",
                "",
                f"**Impact on fraud probability:** {_business_impact_text(feature, row['mean_shap'])}",
                "",
                "**Business interpretation:** When this signal is strong, fraud teams should treat it as a ",
                "**prioritization clue** for manual review or rule design—not as an automatic block on its own.",
                "",
            ]
        )

    lines.extend(
        [
            "## Example case (waterfall plot)",
            "",
            f"The waterfall chart uses explained holdout row **{waterfall_row}** to show how individual ",
            "features raised or lowered the fraud score for one transaction. This is useful when ",
            "an analyst asks, \"Why was this specific order flagged?\"",
            "",
            "## Responsible use",
            "",
            "- Geographic and channel signals should inform **review queues**, not blanket customer rejection.",
            "- SHAP reflects patterns in historical data; new fraud tactics may not appear until retraining.",
            "- Combine model explanations with policy, customer context, and investigator judgment.",
        ]
    )
    return "\n".join(lines)


def save_shap_summary_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    output_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, features, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_bar_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    output_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_waterfall_plot(
    explainer,
    x_explain: pd.DataFrame,
    row_index: int,
    output_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    row = x_explain.iloc[[row_index]]
    shap_values = compute_shap_values(explainer, row)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=row.iloc[0].values,
        feature_names=list(x_explain.columns),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def save_shap_dependence_plot(
    shap_values: np.ndarray,
    features: pd.DataFrame,
    feature_name: str,
    output_path: Path,
) -> Path:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feature_name,
        shap_values,
        features,
        show=False,
        interaction_index=None,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def load_best_model_for_shap(
    *,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> ModelingWorkflowResult:
    """Train/tune models and return the best result with training data for SHAP."""
    log = logger or get_logger(__name__)
    return run_fraud_modeling_workflow(
        save_report=False,
        store_training_data=True,
        random_state=random_state,
        logger=log,
    )


def run_shap_explainability_workflow(
    result: ModelTrainingResult | None = None,
    *,
    workflow: ModelingWorkflowResult | None = None,
    background_size: int = 200,
    explain_size: int = 300,
    top_n: int = 10,
    figures_dir: Path = SHAP_FIGURES_DIR,
    reports_dir: Path = SHAP_REPORTS_DIR,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> ShapAnalysisResult:
    """
    Generate SHAP plots and a stakeholder summary for the best fraud model.

    Pass ``result`` directly, or ``workflow`` from ``run_fraud_modeling_workflow``,
    or leave both unset to train and select the best model automatically.
    """
    log = logger or get_logger(__name__)

    if result is None:
        if workflow is None:
            log.info("No model supplied; training workflow to identify best model")
            workflow = load_best_model_for_shap(random_state=random_state, logger=log)
        result = workflow.best_result

    x_background, x_explain, explain_indices = _resolve_matrices(
        result,
        background_size=background_size,
        explain_size=explain_size,
        random_state=random_state,
    )

    explainer = create_shap_explainer(result, x_background)
    shap_values = compute_shap_values(explainer, x_explain)
    top_features = rank_top_shap_features(
        shap_values,
        result.feature_names,
        top_n=top_n,
    )

    figure_dir = ensure_dir(figures_dir)
    report_dir = ensure_dir(reports_dir)
    local_waterfall_index = _pick_local_waterfall_index(result, explain_indices)
    waterfall_row = int(explain_indices[local_waterfall_index])

    paths = ShapAnalysisPaths(
        output_dir=report_dir,
        figures_dir=figure_dir,
        summary_plot=figure_dir / SUMMARY_PLOT,
        bar_plot=figure_dir / BAR_PLOT,
        waterfall_plot=figure_dir / WATERFALL_PLOT,
        dependence_plot=figure_dir / DEPENDENCE_PLOT,
        top_features_csv=report_dir / TOP_FEATURES_CSV,
        summary_markdown=report_dir / SUMMARY_MD,
    )

    save_shap_summary_plot(shap_values, x_explain, paths.summary_plot)
    save_shap_bar_plot(shap_values, x_explain, paths.bar_plot)
    save_shap_waterfall_plot(explainer, x_explain, local_waterfall_index, paths.waterfall_plot)
    save_shap_dependence_plot(
        shap_values,
        x_explain,
        top_features.iloc[0]["feature"],
        paths.dependence_plot,
    )

    top_features.to_csv(paths.top_features_csv, index=False)
    paths.summary_markdown.write_text(
        build_shap_summary_markdown(result, top_features, waterfall_row=waterfall_row),
        encoding="utf-8",
    )

    log.info("Saved SHAP artifacts for %s", result.model_name)
    log.info("  figures: %s", figure_dir)
    log.info("  summary: %s", paths.summary_markdown)

    return ShapAnalysisResult(
        model_name=result.model_name,
        top_features=top_features,
        shap_values=shap_values,
        explained_features=x_explain,
        paths=paths,
        waterfall_index=waterfall_row,
    )

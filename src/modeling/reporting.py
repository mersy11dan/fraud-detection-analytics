"""Report-ready exports for fraud modeling results."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import (
    MODELING_CONFUSION_DIR,
    MODELING_OUTPUTS_DIR,
    MODELING_PLOTS_DIR,
)
from src.modeling.interpretation import (
    describe_feature,
    extract_feature_importance,
    plot_top_features,
)
from src.modeling.metrics import confusion_matrix_frame
from src.modeling.training import (
    ModelTrainingResult,
    compare_model_results,
    identify_best_model,
)
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

METRICS_FILENAME = "model_comparison_metrics.csv"
BEST_MODEL_SUMMARY_MD = "best_model_summary.md"
BEST_MODEL_SUMMARY_CSV = "best_model_summary.csv"
TOP_FEATURES_FILENAME = "top_features_best_model.csv"
PR_CURVE_PLOT = "model_comparison_pr_curve.png"
ROC_CURVE_PLOT = "model_comparison_roc_curve.png"
FEATURE_IMPORTANCE_PLOT = "best_model_feature_importance.png"

COMPARISON_COLUMNS = [
    "model_name",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives",
]


@dataclass(frozen=True)
class ModelingReportPaths:
    """Paths to report-ready modeling artifacts."""

    output_dir: Path
    metrics_csv: Path
    best_model_summary_md: Path
    best_model_summary_csv: Path
    top_features_csv: Path
    pr_curve_plot: Path
    roc_curve_plot: Path
    feature_importance_plot: Path
    confusion_matrix_plots: tuple[Path, ...]

    def as_dict(self) -> dict[str, Path | tuple[Path, ...]]:
        """Return artifact paths keyed by short name."""
        return asdict(self)


def _safe_filename(model_name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", model_name.lower()).strip("_")


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, separator, *rows])


def format_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return a report-focused comparison table with PR-AUC column."""
    if "pr_auc" not in comparison.columns and "auc_pr" in comparison.columns:
        comparison = comparison.copy()
        comparison["pr_auc"] = comparison["auc_pr"]

    available = [col for col in COMPARISON_COLUMNS if col in comparison.columns]
    return comparison[available].copy()


def build_best_model_summary_markdown(
    comparison: pd.DataFrame,
    best_result: ModelTrainingResult,
    *,
    importance_table: pd.DataFrame | None = None,
    selection_metric: str = "pr_auc",
) -> str:
    """Build a markdown summary of the current best fraud model."""
    best_row = comparison.iloc[0]
    runner_up = comparison.iloc[1] if len(comparison) > 1 else None
    metrics = best_result.metrics
    comparison_view = format_comparison_table(comparison).copy()

    for col in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if col in comparison_view.columns:
            comparison_view[col] = comparison_view[col].map(lambda v: f"{v:.4f}")

    lines = [
        "# Best Fraud Model Summary — Fraud_Data",
        "",
        "## Selected model",
        f"- **Model:** `{best_result.model_name}`",
        f"- **Selection metric:** `{selection_metric}` (higher is better for imbalanced fraud detection)",
        f"- **Holdout PR-AUC:** {metrics.auc_pr:.4f}",
        f"- **Holdout F1:** {metrics.f1:.4f}",
        f"- **Holdout recall:** {metrics.recall:.4f}",
        f"- **Holdout precision:** {metrics.precision:.4f}",
        f"- **Holdout ROC-AUC:** {metrics.roc_auc:.4f}",
        "",
    ]

    if runner_up is not None:
        pr_auc_gap = best_row.get("pr_auc", best_row["auc_pr"]) - runner_up.get(
            "pr_auc", runner_up["auc_pr"]
        )
        f1_gap = best_row["f1"] - runner_up["f1"]
        lines.extend(
            [
                "## Comparison vs runner-up",
                f"- **Runner-up:** `{runner_up['model_name']}`",
                f"- **PR-AUC gap:** +{pr_auc_gap:.4f}",
                f"- **F1 gap:** +{f1_gap:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Model comparison table",
            "",
            _dataframe_to_markdown_table(comparison_view),
            "",
            "## Best model confusion matrix",
            "",
            f"- True positives (fraud caught): **{metrics.true_positives:,}**",
            f"- False negatives (missed fraud): **{metrics.false_negatives:,}**",
            f"- False positives (false alarms): **{metrics.false_positives:,}**",
            f"- True negatives (correctly cleared): **{metrics.true_negatives:,}**",
            "",
        ]
    )

    if importance_table is not None and not importance_table.empty:
        top_features = importance_table.head(5)[["feature", "importance"]].copy()
        top_features["importance"] = top_features["importance"].map(lambda v: f"{v:.4f}")
        lines.extend(
            [
                "## Top feature drivers",
                "",
                _dataframe_to_markdown_table(top_features),
                "",
                "### Plain-language notes",
            ]
        )
        for _, row in importance_table.head(5).iterrows():
            lines.append(f"- `{row['feature']}`: {describe_feature(row['feature'])}")
        lines.append("")

    lines.extend(
        [
            "## Interim recommendation",
            "",
            (
                f"`{best_result.model_name}` is the current best model on the stratified holdout "
                f"when ranked by **PR-AUC**. It should be carried forward for threshold tuning "
                "and deeper explainability work (SHAP), but is not yet production-ready without "
                "those follow-up steps."
            ),
            "",
            "## Report artifacts",
            "",
            f"- `{METRICS_FILENAME}` — full metrics table for all models",
            f"- `{BEST_MODEL_SUMMARY_CSV}` — one-row summary for the best model",
            f"- `{TOP_FEATURES_FILENAME}` — ranked features for the best model",
            f"- `plots/{PR_CURVE_PLOT}` — precision-recall comparison chart",
            f"- `plots/{ROC_CURVE_PLOT}` — ROC comparison chart",
            f"- `confusion_matrices/` — per-model confusion matrix plots",
            f"- `plots/{FEATURE_IMPORTANCE_PLOT}` — top-10 feature importance chart",
        ]
    )
    return "\n".join(lines)


def build_best_model_summary_csv(
    comparison: pd.DataFrame,
    best_result: ModelTrainingResult,
) -> pd.DataFrame:
    """Return a one-row CSV-friendly summary for the best model."""
    best_row = comparison.iloc[0]
    runner_up_name = comparison.iloc[1]["model_name"] if len(comparison) > 1 else None
    runner_up_pr_auc = (
        comparison.iloc[1].get("pr_auc", comparison.iloc[1]["auc_pr"])
        if len(comparison) > 1
        else None
    )
    best_pr_auc = best_row.get("pr_auc", best_row["auc_pr"])

    return pd.DataFrame(
        [
            {
                "model_name": best_result.model_name,
                "runner_up_model": runner_up_name,
                "selection_metric": "pr_auc",
                "precision": best_result.metrics.precision,
                "recall": best_result.metrics.recall,
                "f1": best_result.metrics.f1,
                "roc_auc": best_result.metrics.roc_auc,
                "pr_auc": best_result.metrics.auc_pr,
                "pr_auc_gap_vs_runner_up": (
                    best_pr_auc - runner_up_pr_auc if runner_up_pr_auc is not None else None
                ),
                "true_positives": best_result.metrics.true_positives,
                "false_positives": best_result.metrics.false_positives,
                "false_negatives": best_result.metrics.false_negatives,
                "true_negatives": best_result.metrics.true_negatives,
            }
        ]
    )


def save_model_comparison_pr_curve(
    results: list[ModelTrainingResult],
    output_path: Path,
) -> Path:
    """Save an overlaid precision-recall curve for all models."""
    ensure_dir(output_path.parent)
    baseline_prevalence = results[0].y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_result in results:
        precision_vals, recall_vals, _ = precision_recall_curve(
            model_result.y_true,
            model_result.y_score,
        )
        ax.plot(
            recall_vals,
            precision_vals,
            linewidth=2,
            label=f"{model_result.model_name} (PR-AUC = {model_result.metrics.auc_pr:.3f})",
        )

    ax.axhline(
        baseline_prevalence,
        linestyle="--",
        color="gray",
        label=f"No-skill baseline ({baseline_prevalence:.1%} fraud rate)",
    )
    ax.set_xlabel("Recall (fraud caught)")
    ax.set_ylabel("Precision (flags that are fraud)")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_model_comparison_roc_curve(
    results: list[ModelTrainingResult],
    output_path: Path,
) -> Path:
    """Save an overlaid ROC curve for all models."""
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_result in results:
        fpr, tpr, _ = roc_curve(model_result.y_true, model_result.y_score)
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_result.model_name} (ROC-AUC = {model_result.metrics.roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC Curves — Model Comparison")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_confusion_matrix_plot(
    result: ModelTrainingResult,
    output_path: Path,
) -> Path:
    """Save a confusion matrix heatmap for one model."""
    ensure_dir(output_path.parent)
    cm = confusion_matrix_frame(result.y_true, result.y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Legitimate (0)", "Fraud (1)"],
        yticklabels=["Legitimate (0)", "Fraud (1)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(f"Confusion Matrix — {result.model_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_all_confusion_matrix_plots(
    results: list[ModelTrainingResult],
    output_dir: Path,
) -> list[Path]:
    """Save confusion matrix plots for every evaluated model."""
    ensure_dir(output_dir)
    paths: list[Path] = []
    for result in results:
        path = output_dir / f"confusion_matrix_{_safe_filename(result.model_name)}.png"
        save_confusion_matrix_plot(result, path)
        paths.append(path)
    return paths


def save_feature_importance_plot(
    result: ModelTrainingResult,
    output_path: Path,
    *,
    top_n: int = 10,
) -> Path:
    """Save a top-feature importance chart for the best model."""
    ensure_dir(output_path.parent)
    importance_table = extract_feature_importance(result)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_top_features(importance_table, model_name=result.model_name, top_n=top_n, ax=ax)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_modeling_report(
    results: list[ModelTrainingResult],
    *,
    output_dir: Path = MODELING_OUTPUTS_DIR,
    plots_dir: Path | None = None,
    confusion_dir: Path | None = None,
    top_n_features: int = 20,
    logger: logging.Logger | None = None,
) -> ModelingReportPaths:
    """
    Persist report-ready modeling outputs for interim and final reports.

    Writes metrics CSV, best-model summaries, top features, ROC/PR curves,
    and per-model confusion matrices under ``reports/outputs/`` by default.
    """
    log = logger or get_logger(__name__)
    if not results:
        raise ValueError("At least one model result is required to build a modeling report.")

    report_dir = ensure_dir(output_dir)
    plot_dir = ensure_dir(plots_dir or MODELING_PLOTS_DIR)
    cm_dir = ensure_dir(confusion_dir or MODELING_CONFUSION_DIR)

    comparison = compare_model_results(results, sort_by="auc_pr")
    best_result = identify_best_model(results, metric="auc_pr")
    importance_table = extract_feature_importance(best_result)
    confusion_paths = save_all_confusion_matrix_plots(results, cm_dir)

    paths = ModelingReportPaths(
        output_dir=report_dir,
        metrics_csv=report_dir / METRICS_FILENAME,
        best_model_summary_md=report_dir / BEST_MODEL_SUMMARY_MD,
        best_model_summary_csv=report_dir / BEST_MODEL_SUMMARY_CSV,
        top_features_csv=report_dir / TOP_FEATURES_FILENAME,
        pr_curve_plot=plot_dir / PR_CURVE_PLOT,
        roc_curve_plot=plot_dir / ROC_CURVE_PLOT,
        feature_importance_plot=plot_dir / FEATURE_IMPORTANCE_PLOT,
        confusion_matrix_plots=tuple(confusion_paths),
    )

    format_comparison_table(comparison).to_csv(paths.metrics_csv, index=False)
    paths.best_model_summary_md.write_text(
        build_best_model_summary_markdown(
            comparison,
            best_result,
            importance_table=importance_table,
        ),
        encoding="utf-8",
    )
    build_best_model_summary_csv(comparison, best_result).to_csv(
        paths.best_model_summary_csv,
        index=False,
    )
    importance_table.head(top_n_features).to_csv(paths.top_features_csv, index=False)

    save_model_comparison_pr_curve(results, paths.pr_curve_plot)
    save_model_comparison_roc_curve(results, paths.roc_curve_plot)
    save_feature_importance_plot(best_result, paths.feature_importance_plot)

    log.info("Saved modeling report artifacts to %s", report_dir)
    log.info("Best model: %s (PR-AUC=%.4f)", best_result.model_name, best_result.metrics.auc_pr)
    for name, path in paths.as_dict().items():
        if name not in {"output_dir", "confusion_matrix_plots"}:
            log.info("  %s: %s", name, path)

    return paths

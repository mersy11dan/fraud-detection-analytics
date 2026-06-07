"""Class imbalance handling for fraud detection modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from src.preprocessing.inspect import class_imbalance_summary
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

ResamplingStrategy = Literal["smote", "undersample"]
TARGET_COLUMN = "class"


@dataclass
class ImbalanceHandlingResult:
    """Container for train/test splits and resampling diagnostics."""

    x_train_resampled: pd.DataFrame
    y_train_resampled: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    distribution_comparison: pd.DataFrame
    strategy: ResamplingStrategy
    summary_markdown: str


def _labels_to_frame(y: pd.Series | np.ndarray, stage: str) -> pd.DataFrame:
    """Convert labels to a class distribution summary table."""
    labels = pd.Series(y, name=TARGET_COLUMN)
    summary = class_imbalance_summary(labels.to_frame(), target_column=TARGET_COLUMN)
    summary.insert(0, "stage", stage)
    summary["imbalance_ratio"] = summary.attrs.get("imbalance_ratio")
    return summary


def compare_class_distributions(
    y_train_before: pd.Series,
    y_train_after: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare class distributions before/after resampling and on the holdout test set."""
    frames = [
        _labels_to_frame(y_train_before, "train_before_resampling"),
        _labels_to_frame(y_train_after, "train_after_resampling"),
        _labels_to_frame(y_test, "test_holdout_unmodified"),
    ]
    return pd.concat(frames, ignore_index=True)


def _build_resampler(strategy: ResamplingStrategy, random_state: int):
    """
    Create the configured resampler.

    SMOTE is the default because Fraud_Data has ~9:1 imbalance and undersampling
    would discard most legitimate transactions. SMOTE keeps all majority rows and
    synthesizes minority examples for better fraud recall during training.
    """
    if strategy == "smote":
        return SMOTE(random_state=random_state)
    if strategy == "undersample":
        return RandomUnderSampler(random_state=random_state)
    raise ValueError(f"Unsupported resampling strategy: {strategy}")


def prepare_resampled_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    strategy: ResamplingStrategy = "smote",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> ImbalanceHandlingResult:
    """
    Split data, resample the training set only, and return diagnostics.

    The test set is stratified but never resampled so evaluation reflects
    real-world class prevalence.
    """
    log = logger or get_logger(__name__)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    log.info("Train size before resampling: %s", len(x_train))
    log.info("Test size (holdout, not resampled): %s", len(x_test))
    log.info(
        "Train class distribution before resampling:\n%s",
        _labels_to_frame(y_train, "train_before_resampling"),
    )

    resampler = _build_resampler(strategy, random_state)
    x_resampled, y_resampled = resampler.fit_resample(x_train, y_train)

    x_train_resampled = pd.DataFrame(x_resampled, columns=features.columns)
    y_train_resampled = pd.Series(y_resampled, name=TARGET_COLUMN)

    log.info("Train size after %s: %s", strategy, len(x_train_resampled))
    log.info(
        "Train class distribution after resampling:\n%s",
        _labels_to_frame(y_train_resampled, "train_after_resampling"),
    )
    log.info(
        "Test class distribution (unchanged):\n%s",
        _labels_to_frame(y_test, "test_holdout_unmodified"),
    )

    comparison = compare_class_distributions(y_train, y_train_resampled, y_test)
    summary_markdown = build_interim_summary_markdown(
        comparison,
        strategy=strategy,
        train_size_before=len(x_train),
        train_size_after=len(x_train_resampled),
        test_size=len(x_test),
    )

    return ImbalanceHandlingResult(
        x_train_resampled=x_train_resampled,
        y_train_resampled=y_train_resampled,
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        distribution_comparison=comparison,
        strategy=strategy,
        summary_markdown=summary_markdown,
    )


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a markdown table without extra dependencies."""
    headers = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([headers, separator, *rows])


def build_interim_summary_markdown(
    comparison: pd.DataFrame,
    *,
    strategy: ResamplingStrategy,
    train_size_before: int,
    train_size_after: int,
    test_size: int,
) -> str:
    """Build a concise markdown summary for the interim report."""
    before = comparison[comparison["stage"] == "train_before_resampling"]
    after = comparison[comparison["stage"] == "train_after_resampling"]
    test = comparison[comparison["stage"] == "test_holdout_unmodified"]

    technique_note = (
        "SMOTE synthesizes minority-class examples in feature space while keeping "
        "all legitimate training transactions."
        if strategy == "smote"
        else "Random undersampling reduces majority-class rows to match the minority count."
    )

    lines = [
        "# Class Imbalance Handling Summary",
        "",
        "## Technique",
        f"- **Method:** `{strategy}` on the training set only",
        f"- **Rationale:** {technique_note}",
        "- **Test set:** never resampled (stratified holdout)",
        "",
        "## Split sizes",
        f"- Training rows before resampling: **{train_size_before:,}**",
        f"- Training rows after resampling: **{train_size_after:,}**",
        f"- Test rows (unchanged): **{test_size:,}**",
        "",
        "## Class distribution comparison",
        "",
        "### Training set - before resampling",
        _dataframe_to_markdown_table(before[["class", "count", "pct"]].reset_index(drop=True)),
        "",
        "### Training set - after resampling",
        _dataframe_to_markdown_table(after[["class", "count", "pct"]].reset_index(drop=True)),
        "",
        "### Test set - holdout (unmodified)",
        _dataframe_to_markdown_table(test[["class", "count", "pct"]].reset_index(drop=True)),
        "",
        "## Interim takeaway",
        (
            "Resampling is applied only to training data so evaluation on the holdout "
            "set still reflects natural fraud prevalence. SMOTE improves the model's "
            "exposure to rare fraud patterns without discarding legitimate transactions."
            if strategy == "smote"
            else "Undersampling creates a balanced training set by removing majority "
            "rows; use when training speed is prioritized over retaining all legitimate data."
        ),
    ]
    return "\n".join(lines)


def save_imbalance_report(
    result: ImbalanceHandlingResult,
    *,
    output_dir: Path = REPORTS_DIR,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Persist comparison tables and interim markdown summary."""
    log = logger or get_logger(__name__)
    report_dir = ensure_dir(output_dir)

    comparison_path = report_dir / "class_imbalance_comparison.csv"
    summary_path = report_dir / "class_imbalance_summary.md"

    result.distribution_comparison.to_csv(comparison_path, index=False)
    summary_path.write_text(result.summary_markdown, encoding="utf-8")

    log.info("Saved class distribution comparison to %s", comparison_path)
    log.info("Saved interim imbalance summary to %s", summary_path)
    return {"comparison_csv": comparison_path, "summary_md": summary_path}

"""Tests for report-ready modeling exports."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.reporting import (
    build_best_model_summary_csv,
    build_best_model_summary_markdown,
    save_modeling_report,
)
from src.modeling.training import compare_model_results, train_classifier


def _tiny_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
    y_train = pd.Series([0] * 45 + [1] * 15, name="class")
    x_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=["a", "b", "c", "d"])
    y_test = pd.Series([0] * 16 + [1] * 4, name="class")
    return x_train, y_train, x_test, y_test


def test_build_best_model_summary_markdown_contains_key_sections() -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    first = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="logistic_regression",
    )
    comparison = compare_model_results([first], sort_by="auc_pr")
    markdown = build_best_model_summary_markdown(comparison, first)

    assert "# Best Fraud Model Summary" in markdown
    assert "logistic_regression" in markdown
    assert "Model comparison table" in markdown


def test_save_modeling_report_writes_expected_artifacts(tmp_path: Path) -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    first = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="logistic_regression",
    )
    second = train_classifier(
        RandomForestClassifier(n_estimators=20, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="random_forest_tuned",
    )

    paths = save_modeling_report([first, second], output_dir=tmp_path)

    assert paths.metrics_csv.exists()
    assert paths.best_model_summary_md.exists()
    assert paths.best_model_summary_csv.exists()
    assert paths.top_features_csv.exists()
    assert paths.pr_curve_plot.exists()
    assert paths.confusion_matrix_plot.exists()
    assert paths.feature_importance_plot.exists()

    metrics = pd.read_csv(paths.metrics_csv)
    assert set(metrics["model_name"]) == {"logistic_regression", "random_forest_tuned"}


def test_build_best_model_summary_csv_has_runner_up_gap() -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    first = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="logistic_regression",
    )
    second = train_classifier(
        RandomForestClassifier(n_estimators=20, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="random_forest_tuned",
    )
    comparison = compare_model_results([first, second], sort_by="auc_pr")
    best = next(r for r in [first, second] if r.model_name == comparison.iloc[0]["model_name"])
    summary = build_best_model_summary_csv(comparison, best)

    assert summary.loc[0, "model_name"] == comparison.iloc[0]["model_name"]
    assert summary.loc[0, "auc_pr_gap_vs_runner_up"] == pytest.approx(
        comparison.iloc[0]["auc_pr"] - comparison.iloc[1]["auc_pr"]
    )

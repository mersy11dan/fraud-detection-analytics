"""Tests for SHAP explainability workflow."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.modeling.shap_explain import (
    build_shap_summary_markdown,
    compute_shap_values,
    create_shap_explainer,
    rank_top_shap_features,
    run_shap_explainability_workflow,
)
from src.modeling.metrics import ClassificationMetrics
from src.modeling.training import ModelTrainingResult, train_classifier


def _tiny_result() -> ModelTrainingResult:
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(rng.normal(size=(80, 5)), columns=[f"f{i}" for i in range(5)])
    y_train = pd.Series([0] * 60 + [1] * 20, name="class")
    x_test = pd.DataFrame(rng.normal(size=(30, 5)), columns=[f"f{i}" for i in range(5)])
    y_test = pd.Series([0] * 24 + [1] * 6, name="class")
    return train_classifier(
        RandomForestClassifier(n_estimators=30, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="random_forest",
        store_training_data=True,
    )


def test_rank_top_shap_features_returns_top_n() -> None:
    shap_values = np.array(
        [
            [0.5, 0.1, 0.0, 0.2, 0.0],
            [0.4, 0.2, 0.0, 0.1, 0.0],
            [-0.3, 0.1, 0.0, 0.0, 0.0],
        ]
    )
    features = [f"f{i}" for i in range(5)]
    ranked = rank_top_shap_features(shap_values, features, top_n=3)

    assert len(ranked) == 3
    assert ranked.iloc[0]["feature"] == "f0"


def test_build_shap_summary_markdown_contains_business_sections() -> None:
    result = _tiny_result()
    top_features = pd.DataFrame(
        {
            "feature": ["f0", "f1"],
            "mean_abs_shap": [0.4, 0.2],
            "mean_shap": [0.3, -0.1],
            "impact_direction": ["increases_fraud_risk", "decreases_fraud_risk"],
        }
    )
    markdown = build_shap_summary_markdown(result, top_features, waterfall_row=0)

    assert "SHAP Explainability Summary" in markdown
    assert "Why it matters" in markdown
    assert "Business interpretation" in markdown


def test_run_shap_explainability_workflow_writes_artifacts(tmp_path: Path) -> None:
    result = _tiny_result()
    analysis = run_shap_explainability_workflow(
        result,
        background_size=40,
        explain_size=20,
        top_n=5,
        figures_dir=tmp_path / "figures",
        reports_dir=tmp_path / "shap",
    )

    assert analysis.paths.summary_plot.exists()
    assert analysis.paths.bar_plot.exists()
    assert analysis.paths.waterfall_plot.exists()
    assert analysis.paths.dependence_plot.exists()
    assert analysis.paths.top_features_csv.exists()
    assert analysis.paths.summary_markdown.exists()
    assert len(analysis.top_features) == 5


def test_compute_shap_values_for_tree_model() -> None:
    result = _tiny_result()
    x_background = result.x_train.iloc[:30]
    x_explain = result.x_test.iloc[:10]
    explainer = create_shap_explainer(result, x_background)
    values = compute_shap_values(explainer, x_explain)

    assert values.shape == (len(x_explain), len(result.feature_names))

"""Tests for the end-to-end modeling workflow."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.reporting import save_modeling_report
from src.modeling.training import train_classifier
from src.modeling.workflow import run_fraud_modeling_workflow


def _patch_workflow_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use tiny estimators so workflow tests stay fast."""

    def fast_configs(*, random_state: int = 42):
        return {
            "logistic_regression": {
                "estimator": LogisticRegression(max_iter=200, random_state=random_state),
                "search": "grid",
                "param_grid": {"C": [0.1, 1.0]},
            },
            "random_forest": {
                "estimator": RandomForestClassifier(random_state=random_state, n_jobs=1),
                "search": "grid",
                "param_grid": {"n_estimators": [10], "max_depth": [3]},
            },
            "xgboost": {
                "estimator": __import__("xgboost").XGBClassifier(
                    random_state=random_state,
                    eval_metric="logloss",
                    n_jobs=1,
                ),
                "search": "grid",
                "param_grid": {"n_estimators": [10], "max_depth": [3]},
            },
        }

    monkeypatch.setattr("src.modeling.workflow.get_tuning_configs", fast_configs)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "fraud_data_features.csv").exists(),
    reason="Processed fraud feature matrix not available",
)
def test_run_fraud_modeling_workflow_returns_three_models(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_workflow_tuning(monkeypatch)
    workflow = run_fraud_modeling_workflow(save_report=False, cv=2)

    assert len(workflow.results) == 3
    assert set(workflow.comparison["model_name"]) == {
        "logistic_regression",
        "random_forest",
        "xgboost",
    }
    assert workflow.best_result.model_name in workflow.comparison.iloc[0]["model_name"]
    assert workflow.dataset_name == "Fraud_Data"
    assert "pr_auc" in workflow.comparison.columns


def test_save_modeling_report_writes_roc_and_all_confusion_matrices(tmp_path: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(rng.normal(size=(60, 4)), columns=["a", "b", "c", "d"])
    y_train = pd.Series([0] * 45 + [1] * 15, name="class")
    x_test = pd.DataFrame(rng.normal(size=(20, 4)), columns=["a", "b", "c", "d"])
    y_test = pd.Series([0] * 16 + [1] * 4, name="class")

    results = [
        train_classifier(
            LogisticRegression(max_iter=500, random_state=42),
            x_train,
            y_train,
            x_test,
            y_test,
            model_name="logistic_regression",
        ),
        train_classifier(
            RandomForestClassifier(n_estimators=20, random_state=42),
            x_train,
            y_train,
            x_test,
            y_test,
            model_name="random_forest",
        ),
    ]

    paths = save_modeling_report(results, output_dir=tmp_path)

    assert paths.roc_curve_plot.exists()
    assert len(paths.confusion_matrix_plots) == 2
    metrics = pd.read_csv(paths.metrics_csv)
    assert "pr_auc" in metrics.columns

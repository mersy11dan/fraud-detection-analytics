"""Tests for hyperparameter tuning."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling.tuning import get_tuning_configs, tune_classifier


def _tiny_train() -> tuple[pd.DataFrame, pd.Series]:
    import numpy as np

    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(rng.normal(size=(50, 3)), columns=["a", "b", "c"])
    y_train = pd.Series([0] * 40 + [1] * 10, name="class")
    return x_train, y_train


def test_get_tuning_configs_includes_three_models() -> None:
    configs = get_tuning_configs(random_state=42)
    assert set(configs) == {"logistic_regression", "random_forest", "xgboost"}


def test_tune_classifier_logistic_regression_grid() -> None:
    x_train, y_train = _tiny_train()
    config = {
        "estimator": LogisticRegression(max_iter=500, random_state=42),
        "search": "grid",
        "param_grid": {"C": [0.1, 1.0]},
    }
    result = tune_classifier(
        "logistic_regression",
        x_train,
        y_train,
        config=config,
        cv=2,
    )

    assert result.model_name == "logistic_regression"
    assert result.search_strategy == "grid"
    assert "C" in result.best_params
    assert isinstance(result.best_estimator, LogisticRegression)

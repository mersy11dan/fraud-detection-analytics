"""Tests for feature importance interpretation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.interpretation import (
    extract_feature_importance,
    get_coefficient_extremes,
    plot_top_features,
    save_top_features,
)
from src.modeling.training import ModelTrainingResult, train_classifier


def _tiny_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x_train = pd.DataFrame(rng.normal(size=(40, 4)), columns=["a", "b", "c", "d"])
    y_train = pd.Series([0] * 30 + [1] * 10, name="class")
    x_test = pd.DataFrame(rng.normal(size=(10, 4)), columns=["a", "b", "c", "d"])
    y_test = pd.Series([0] * 8 + [1] * 2, name="class")
    return x_train, y_train, x_test, y_test


def test_extract_feature_importance_for_logistic_regression() -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    result = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="logistic_regression",
    )

    importance = extract_feature_importance(result)

    assert list(importance.columns) == [
        "feature",
        "coefficient",
        "importance",
        "direction",
        "importance_type",
    ]
    assert importance["importance_type"].eq("coefficient").all()
    assert importance["importance"].is_monotonic_decreasing


def test_get_coefficient_extremes_returns_signed_groups() -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    result = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
    )
    importance = extract_feature_importance(result)
    positive, negative = get_coefficient_extremes(importance, top_n=2)

    assert (positive["coefficient"] > 0).all()
    assert (negative["coefficient"] < 0).all()


def test_extract_feature_importance_for_random_forest() -> None:
    x_train, y_train, x_test, y_test = _tiny_split()
    result = train_classifier(
        RandomForestClassifier(n_estimators=20, random_state=42),
        x_train,
        y_train,
        x_test,
        y_test,
        model_name="random_forest",
    )

    importance = extract_feature_importance(result)

    assert importance["importance_type"].eq("feature_importance").all()
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_save_top_features_writes_csv(tmp_path) -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b"],
            "importance": [0.7, 0.3],
            "coefficient": [np.nan, np.nan],
            "direction": ["n/a", "n/a"],
            "importance_type": ["feature_importance", "feature_importance"],
        }
    )
    path = save_top_features(importance, model_name="random_forest", output_dir=tmp_path, top_n=2)
    saved = pd.read_csv(path)

    assert path.exists()
    assert len(saved) == 2


def test_plot_top_features_returns_axes() -> None:
    importance = pd.DataFrame(
        {
            "feature": ["a", "b", "c"],
            "importance": [0.5, 0.3, 0.2],
            "coefficient": [np.nan, np.nan, np.nan],
            "direction": ["n/a", "n/a", "n/a"],
            "importance_type": ["feature_importance", "feature_importance", "feature_importance"],
        }
    )
    ax = plot_top_features(importance, model_name="random_forest", top_n=3)
    assert ax.get_title().startswith("Top Feature Importances")

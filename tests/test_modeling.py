"""Tests for fraud classification modeling utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.data import (
    ModelingSplit,
    load_creditcard_feature_matrix,
    load_fraud_feature_matrix,
    stratified_train_test_split,
)
from src.modeling.metrics import compute_classification_metrics, confusion_matrix_frame
from src.modeling.training import (
    compare_model_results,
    get_default_estimators,
    save_model_metrics,
    train_classifier,
    train_multiple_classifiers,
)


def _synthetic_dataset(n_majority: int = 80, n_minority: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x_majority = rng.normal(size=(n_majority, 4))
    x_minority = rng.normal(loc=1.5, size=(n_minority, 4))
    features = pd.DataFrame(
        np.vstack([x_majority, x_minority]),
        columns=["f1", "f2", "f3", "f4"],
    )
    target = pd.Series([0] * n_majority + [1] * n_minority, name="class")
    return features, target


def test_stratified_train_test_split_preserves_minority_class() -> None:
    features, target = _synthetic_dataset()
    split = stratified_train_test_split(features, target, test_size=0.2, random_state=42)

    assert isinstance(split, ModelingSplit)
    assert len(split.x_train) + len(split.x_test) == len(features)
    assert split.feature_names == ["f1", "f2", "f3", "f4"]
    assert split.y_test.value_counts()[1] > 0
    assert split.y_train.value_counts()[1] > 0


def test_compute_classification_metrics_returns_expected_fields() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.6, 0.8, 0.4])

    metrics = compute_classification_metrics(y_true, y_pred, y_score)

    assert metrics.accuracy == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert 0.0 <= metrics.roc_auc <= 1.0
    assert 0.0 <= metrics.auc_pr <= 1.0
    assert metrics.true_positives + metrics.false_negatives == 2
    assert metrics.true_negatives + metrics.false_positives == 2


def test_confusion_matrix_frame_has_labeled_axes() -> None:
    frame = confusion_matrix_frame([0, 1, 1], [0, 0, 1])
    assert list(frame.index) == ["actual_0", "actual_1"]
    assert list(frame.columns) == ["predicted_0", "predicted_1"]


def test_train_classifier_returns_fitted_model_and_scores() -> None:
    features, target = _synthetic_dataset()
    split = stratified_train_test_split(features, target, test_size=0.2, random_state=42)

    result = train_classifier(
        LogisticRegression(max_iter=500, random_state=42),
        split.x_train,
        split.y_train,
        split.x_test,
        split.y_test,
        model_name="logistic_regression",
        store_training_data=True,
    )

    assert result.model_name == "logistic_regression"
    assert hasattr(result.estimator, "predict_proba")
    assert len(result.y_pred) == len(split.y_test)
    assert len(result.y_score) == len(split.y_test)
    assert result.x_train is not None
    assert result.feature_names == split.feature_names


def test_train_classifier_accepts_default_model_name() -> None:
    features, target = _synthetic_dataset()
    split = stratified_train_test_split(features, target, test_size=0.2, random_state=42)

    result = train_classifier(
        "logistic_regression",
        split.x_train,
        split.y_train,
        split.x_test,
        split.y_test,
        random_state=42,
    )

    assert result.model_name == "logistic_regression"
    assert result.metrics.f1 >= 0.0


def test_train_multiple_classifiers_and_compare_results() -> None:
    features, target = _synthetic_dataset()
    split = stratified_train_test_split(features, target, test_size=0.2, random_state=42)

    results = train_multiple_classifiers(
        ["logistic_regression", "random_forest"],
        split.x_train,
        split.y_train,
        split.x_test,
        split.y_test,
        random_state=42,
    )
    comparison = compare_model_results(results, sort_by="f1")

    assert len(results) == 2
    assert set(comparison["model_name"]) == {"logistic_regression", "random_forest"}
    assert "auc_pr" in comparison.columns
    assert "true_positives" in comparison.columns


def test_save_model_metrics_writes_csv(tmp_path: Path) -> None:
    features, target = _synthetic_dataset()
    split = stratified_train_test_split(features, target, test_size=0.2, random_state=42)
    result = train_classifier(
        DummyClassifier(strategy="most_frequent"),
        split.x_train,
        split.y_train,
        split.x_test,
        split.y_test,
        model_name="dummy",
    )

    output_path = save_model_metrics([result], output_dir=tmp_path)
    saved = pd.read_csv(output_path)

    assert output_path.exists()
    assert saved.loc[0, "model_name"] == "dummy"
    assert "recall" in saved.columns


def test_get_default_estimators_includes_baselines() -> None:
    estimators = get_default_estimators(random_state=42)
    assert set(estimators) == {"logistic_regression", "random_forest", "xgboost"}


def test_load_creditcard_feature_matrix_from_synthetic_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    rows = 200
    fraud_count = 20
    frame = pd.DataFrame(
        {
            "time": rng.uniform(0, 100000, rows),
            "amount": rng.uniform(1, 200, rows),
            "class": [1] * fraud_count + [0] * (rows - fraud_count),
        }
    )
    for idx in range(1, 29):
        frame[f"v{idx}"] = rng.normal(size=rows)

    frame = frame.sample(frac=1, random_state=42).reset_index(drop=True)
    frame.to_csv(tmp_path / "creditcard_clean.csv", index=False)

    features, target = load_creditcard_feature_matrix(data_dir=tmp_path)

    assert len(features) == rows
    assert len(features.columns) == 30
    assert target.sum() == fraud_count


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data" / "processed" / "fraud_data_features.csv").exists(),
    reason="Processed fraud feature matrix not available",
)
def test_load_fraud_feature_matrix_from_disk() -> None:
    features, target = load_fraud_feature_matrix()
    assert "class" not in features.columns
    assert target.name == "class"
    assert len(features) == len(target)
    assert len(features.columns) > 0

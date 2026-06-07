"""Tests for class imbalance handling."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling.imbalance import (
    compare_class_distributions,
    prepare_resampled_training_data,
    save_imbalance_report,
)


def _imbalanced_feature_matrix(n_majority: int = 90, n_minority: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    x_majority = rng.normal(size=(n_majority, 3))
    x_minority = rng.normal(loc=2.0, size=(n_minority, 3))
    features = pd.DataFrame(
        np.vstack([x_majority, x_minority]),
        columns=["f1", "f2", "f3"],
    )
    target = pd.Series([0] * n_majority + [1] * n_minority, name="class")
    return features, target


def test_prepare_resampled_training_data_smote_balances_train_only() -> None:
    features, target = _imbalanced_feature_matrix()
    result = prepare_resampled_training_data(features, target, strategy="smote", test_size=0.2)

    train_after = result.y_train_resampled.value_counts()
    assert train_after[0] == train_after[1]

    test_counts = result.y_test.value_counts()
    assert len(result.x_test) == len(result.y_test)
    assert len(result.x_test) == pytest.approx(20, rel=0.2)
    assert test_counts[0] > test_counts[1]


def test_test_set_is_never_resampled() -> None:
    features, target = _imbalanced_feature_matrix()
    result = prepare_resampled_training_data(features, target, strategy="smote", test_size=0.2)

    expected_test_size = int(len(features) * 0.2)
    assert len(result.x_test) == expected_test_size
    assert result.distribution_comparison.loc[
        result.distribution_comparison["stage"] == "test_holdout_unmodified", "count"
    ].sum() == expected_test_size


def test_compare_class_distributions_includes_all_stages() -> None:
    features, target = _imbalanced_feature_matrix()
    result = prepare_resampled_training_data(features, target, strategy="undersample", test_size=0.2)

    stages = set(result.distribution_comparison["stage"])
    assert stages == {
        "train_before_resampling",
        "train_after_resampling",
        "test_holdout_unmodified",
    }


def test_save_imbalance_report_writes_files(tmp_path: Path) -> None:
    features, target = _imbalanced_feature_matrix(n_majority=40, n_minority=10)
    result = prepare_resampled_training_data(features, target, strategy="undersample", test_size=0.2)
    paths = save_imbalance_report(result, output_dir=tmp_path)

    assert paths["comparison_csv"].exists()
    assert paths["summary_md"].exists()
    assert "Class Imbalance Handling Summary" in paths["summary_md"].read_text(encoding="utf-8")

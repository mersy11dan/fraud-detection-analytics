"""Tests for preprocessing utilities."""

import pandas as pd
import pytest

from src.preprocessing.pipeline import split_features_target


def test_split_features_target() -> None:
    df = pd.DataFrame({"feature_a": [1, 2, 3], "feature_b": [4, 5, 6], "Class": [0, 1, 0]})
    X, y = split_features_target(df, target_column="Class")

    assert list(X.columns) == ["feature_a", "feature_b"]
    assert y.tolist() == [0, 1, 0]


def test_split_features_target_missing_column() -> None:
    df = pd.DataFrame({"feature_a": [1, 2, 3]})

    with pytest.raises(ValueError, match="Target column 'Class' not found"):
        split_features_target(df, target_column="Class")

"""Tests for inspection helpers."""

import pandas as pd
import pytest

from src.preprocessing.inspect import (
    class_imbalance_summary,
    duplicate_check,
    missing_values_summary,
    summary_statistics,
)


def test_missing_values_summary() -> None:
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    summary = missing_values_summary(df)

    assert summary.loc["a", "missing_count"] == 1
    assert summary.loc["b", "missing_count"] == 1


def test_duplicate_check() -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 10, 20]})
    result = duplicate_check(df)

    assert result["duplicate_rows_to_remove"] == 1
    assert result["rows_in_duplicate_groups"] == 2


def test_class_imbalance_summary() -> None:
    df = pd.DataFrame({"class": [0, 0, 0, 1]})
    summary = class_imbalance_summary(df, target_column="class")

    assert summary["count"].tolist() == [3, 1]
    assert summary.attrs["imbalance_ratio"] == 3.0


def test_class_imbalance_summary_missing_column() -> None:
    df = pd.DataFrame({"feature": [1, 2, 3]})

    with pytest.raises(ValueError, match="Target column 'class' not found"):
        class_imbalance_summary(df, target_column="class")


def test_summary_statistics_includes_numeric_and_categorical() -> None:
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0], "source": ["a", "b", "a"]})
    summary = summary_statistics(df)

    assert "amount" in summary.index
    assert "source" in summary.index

"""Tests for cleaning utilities."""

import pandas as pd

from src.preprocessing.cleaning import (
    convert_timestamps,
    handle_duplicates,
    handle_missing_values,
)
from src.preprocessing.columns import standardize_column_names


def test_standardize_column_names() -> None:
    df = pd.DataFrame({"Purchase Time": [1], "Class": [0]})
    renamed = standardize_column_names(df)

    assert list(renamed.columns) == ["purchase_time", "class"]


def test_handle_missing_values_fills_numeric_and_categorical() -> None:
    df = pd.DataFrame({"age": [20, None, 30], "browser": ["Chrome", None, "Safari"]})
    cleaned = handle_missing_values(df, numeric_fill="median", categorical_fill="unknown")

    assert cleaned["age"].tolist() == [20, 25, 30]
    assert cleaned["browser"].tolist() == ["Chrome", "unknown", "Safari"]


def test_handle_duplicates_removes_rows() -> None:
    df = pd.DataFrame({"id": [1, 1, 2], "value": [10, 10, 20]})
    cleaned = handle_duplicates(df)

    assert len(cleaned) == 2


def test_convert_timestamps() -> None:
    df = pd.DataFrame({"purchase_time": ["2015-01-01 10:00:00", "2015-01-02 11:00:00"]})
    converted = convert_timestamps(df, columns=["purchase_time"])

    assert pd.api.types.is_datetime64_any_dtype(converted["purchase_time"])

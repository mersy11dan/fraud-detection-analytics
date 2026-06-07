"""Reusable preprocessing helpers."""

from typing import Tuple

import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into feature matrix X and target vector y."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

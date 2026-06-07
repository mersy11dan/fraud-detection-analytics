"""Column name standardization utilities."""

import re

import pandas as pd


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case."""
    renamed = df.copy()
    renamed.columns = [_to_snake_case(col) for col in renamed.columns]
    return renamed


def _to_snake_case(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned.lower()

"""Data cleaning utilities for missing values, duplicates, and dtypes."""

import logging
from typing import Literal

import pandas as pd

from src.utils.logging_config import get_logger

FillStrategy = Literal["median", "mean", "zero", "mode", "unknown"]


def handle_missing_values(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    numeric_fill: FillStrategy = "median",
    categorical_fill: FillStrategy = "mode",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Handle missing values with logging and sensible defaults per dtype."""
    log = logger or get_logger(__name__)
    cleaned = df.copy()

    missing_before = int(cleaned.isna().sum().sum())
    if missing_before == 0:
        log.info("No missing values found.")
        return cleaned

    log.info("Missing values before cleaning: %s", missing_before)
    for column, count in cleaned.isna().sum().items():
        if count:
            log.info("  %s: %s missing", column, count)

    if required_columns:
        missing_required = [col for col in required_columns if col not in cleaned.columns]
        if missing_required:
            raise ValueError(f"Required columns not found: {missing_required}")

        before_rows = len(cleaned)
        cleaned = cleaned.dropna(subset=required_columns)
        dropped = before_rows - len(cleaned)
        if dropped:
            log.info("Dropped %s rows with missing required columns: %s", dropped, required_columns)

    numeric_cols = cleaned.select_dtypes(include="number").columns
    for column in numeric_cols:
        if cleaned[column].isna().any():
            fill_value = _numeric_fill_value(cleaned[column], numeric_fill)
            cleaned[column] = cleaned[column].fillna(fill_value)
            log.info("Filled numeric column '%s' with %s (%s)", column, numeric_fill, fill_value)

    categorical_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical_cols:
        if cleaned[column].isna().any():
            fill_value = _categorical_fill_value(cleaned[column], categorical_fill)
            cleaned[column] = cleaned[column].fillna(fill_value)
            log.info("Filled categorical column '%s' with %s (%s)", column, categorical_fill, fill_value)

    missing_after = int(cleaned.isna().sum().sum())
    log.info("Missing values after cleaning: %s", missing_after)
    return cleaned


def handle_duplicates(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
    keep: Literal["first", "last", False] = "first",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Remove duplicate rows and log how many were dropped."""
    log = logger or get_logger(__name__)
    before = len(df)
    deduped = df.drop_duplicates(subset=subset, keep=keep)
    removed = before - len(deduped)

    if removed:
        log.info(
            "Removed %s duplicate rows (subset=%s, keep=%s)",
            removed,
            subset,
            keep,
        )
    else:
        log.info("No duplicate rows found (subset=%s).", subset)

    return deduped.reset_index(drop=True)


def convert_timestamps(
    df: pd.DataFrame,
    columns: list[str],
    *,
    errors: Literal["raise", "coerce"] = "coerce",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Parse string columns into pandas datetime dtypes."""
    log = logger or get_logger(__name__)
    converted = df.copy()

    for column in columns:
        if column not in converted.columns:
            raise ValueError(f"Timestamp column '{column}' not found in dataframe.")

        converted[column] = pd.to_datetime(converted[column], errors=errors, utc=False)
        invalid = int(converted[column].isna().sum())
        log.info("Converted '%s' to datetime (%s invalid values)", column, invalid)

    return converted


def _numeric_fill_value(series: pd.Series, strategy: FillStrategy) -> float:
    if strategy == "median":
        return float(series.median())
    if strategy == "mean":
        return float(series.mean())
    if strategy == "zero":
        return 0.0
    raise ValueError(f"Unsupported numeric fill strategy: {strategy}")


def _categorical_fill_value(series: pd.Series, strategy: FillStrategy) -> str:
    if strategy == "mode":
        mode = series.mode(dropna=True)
        return str(mode.iloc[0]) if not mode.empty else "unknown"
    if strategy == "unknown":
        return "unknown"
    raise ValueError(f"Unsupported categorical fill strategy: {strategy}")

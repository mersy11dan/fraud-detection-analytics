"""Temporal feature engineering for fraud transactions."""

import pandas as pd


def add_time_since_signup(
    df: pd.DataFrame,
    *,
    signup_column: str = "signup_time",
    purchase_column: str = "purchase_time",
    output_column: str = "time_since_signup_hours",
) -> pd.DataFrame:
    """
    Compute elapsed hours between account signup and purchase.

    Short signup-to-purchase intervals are a common fraud signal in this dataset.
    """
    enriched = df.copy()
    for column in (signup_column, purchase_column):
        if column not in enriched.columns:
            raise ValueError(f"Required column '{column}' not found.")

    enriched[output_column] = (
        enriched[purchase_column] - enriched[signup_column]
    ).dt.total_seconds() / 3600
    return enriched


def add_purchase_time_features(
    df: pd.DataFrame,
    *,
    purchase_column: str = "purchase_time",
    hour_column: str = "hour_of_day",
    day_column: str = "day_of_week",
) -> pd.DataFrame:
    """Extract hour-of-day and day-of-week from the purchase timestamp."""
    enriched = df.copy()
    if purchase_column not in enriched.columns:
        raise ValueError(f"Required column '{purchase_column}' not found.")

    enriched[hour_column] = enriched[purchase_column].dt.hour.astype("int64")
    enriched[day_column] = enriched[purchase_column].dt.dayofweek.astype("int64")
    return enriched

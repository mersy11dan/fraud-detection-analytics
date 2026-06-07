"""User-level transaction frequency and velocity features."""

import numpy as np
import pandas as pd

DEFAULT_VELOCITY_WINDOWS_HOURS = (1, 24, 168)


def _rolling_txn_count_in_window(
    purchase_times: np.ndarray,
    window_hours: int,
) -> np.ndarray:
    """Count transactions in a trailing time window for one user's timeline."""
    counts = np.zeros(len(purchase_times), dtype=np.int64)
    window = np.timedelta64(window_hours, "h")

    for idx, current_time in enumerate(purchase_times):
        start_time = current_time - window
        counts[idx] = int(np.sum((purchase_times >= start_time) & (purchase_times <= current_time)))

    return counts


def add_user_velocity_features(
    df: pd.DataFrame,
    *,
    user_column: str = "user_id",
    time_column: str = "purchase_time",
    windows_hours: tuple[int, ...] = DEFAULT_VELOCITY_WINDOWS_HOURS,
) -> pd.DataFrame:
    """
    Add per-user transaction frequency and velocity features.

    Features created
    ----------------
    - ``txn_count_last_{N}h``: number of user transactions in the last N hours
      (rolling window ending at the current purchase, inclusive).
    - ``hours_since_last_txn``: hours elapsed since the user's previous purchase.
    - ``user_cumulative_txn_count``: running count of transactions for the user.
    - ``user_txn_velocity_per_day``: cumulative transactions divided by active days
      since the user's first observed purchase.
    """
    required = {user_column, time_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns not found: {sorted(missing)}")

    enriched = df.copy()
    enriched["_row_order"] = np.arange(len(enriched))
    enriched = enriched.sort_values([user_column, time_column])

    enriched["hours_since_last_txn"] = (
        enriched.groupby(user_column, sort=False)[time_column]
        .diff()
        .dt.total_seconds()
        .div(3600)
        .fillna(0.0)
    )
    enriched["user_cumulative_txn_count"] = (
        enriched.groupby(user_column, sort=False).cumcount() + 1
    ).astype("int64")

    first_purchase = enriched.groupby(user_column, sort=False)[time_column].transform("min")
    last_purchase = enriched.groupby(user_column, sort=False)[time_column].transform("max")
    active_days = (last_purchase - first_purchase).dt.total_seconds() / 86400
    enriched["user_txn_velocity_per_day"] = enriched["user_cumulative_txn_count"] / active_days.clip(
        lower=1 / 24
    )

    user_txn_counts = enriched.groupby(user_column, sort=False)[time_column].transform("size")
    multi_txn_mask = user_txn_counts > 1

    for hours in windows_hours:
        column = f"txn_count_last_{hours}h"
        enriched[column] = 1

        if multi_txn_mask.any():
            repeat_users = enriched.loc[multi_txn_mask]
            for _, group in repeat_users.groupby(user_column, sort=False):
                user_times = group[time_column].to_numpy(dtype="datetime64[ns]")
                user_counts = _rolling_txn_count_in_window(user_times, hours)
                enriched.loc[group.index, column] = user_counts

        enriched[column] = enriched[column].astype("int64")

    return enriched.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)

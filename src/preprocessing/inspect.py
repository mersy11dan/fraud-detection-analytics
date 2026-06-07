"""Data inspection and profiling helpers."""

from typing import Any

import pandas as pd


def summary_statistics(
    df: pd.DataFrame,
    *,
    include_categorical: bool = True,
) -> pd.DataFrame:
    """Return descriptive statistics for numeric and optional categorical columns."""
    frames: list[pd.DataFrame] = []

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        numeric_stats = numeric.describe().T
        numeric_stats["dtype"] = numeric.dtypes.astype(str).values
        numeric_stats["missing"] = numeric.isna().sum().values
        frames.append(numeric_stats)

    if include_categorical:
        categorical = df.select_dtypes(include=["object", "string", "category"])
        if not categorical.empty:
            cat_rows = []
            for column in categorical.columns:
                series = categorical[column]
                cat_rows.append(
                    {
                        "column": column,
                        "dtype": str(series.dtype),
                        "count": int(series.count()),
                        "missing": int(series.isna().sum()),
                        "unique": int(series.nunique(dropna=True)),
                        "top": series.mode(dropna=True).iloc[0] if series.notna().any() else None,
                        "freq": int(series.value_counts(dropna=True).iloc[0])
                        if series.notna().any()
                        else 0,
                    }
                )
            frames.append(pd.DataFrame(cat_rows).set_index("column"))

    if not frames:
        return pd.DataFrame()

    if len(frames) == 1:
        return frames[0]

    return pd.concat(frames, axis=0, sort=False)


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize missing values per column."""
    missing_count = df.isna().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "dtype": df.dtypes.astype(str),
        }
    )
    return summary.sort_values("missing_count", ascending=False)


def duplicate_check(
    df: pd.DataFrame,
    *,
    subset: list[str] | None = None,
) -> dict[str, Any]:
    """Count duplicate rows and return a small summary dictionary."""
    duplicate_mask = df.duplicated(subset=subset, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    unique_duplicate_keys = int(df.duplicated(subset=subset).sum())

    return {
        "rows_in_duplicate_groups": duplicate_rows,
        "duplicate_rows_to_remove": unique_duplicate_keys,
        "duplicate_pct": round(unique_duplicate_keys / len(df) * 100, 4) if len(df) else 0.0,
        "subset": subset,
    }


def class_imbalance_summary(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Summarize target class distribution for imbalanced classification tasks."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    counts = df[target_column].value_counts(dropna=False).sort_index()
    total = int(counts.sum())
    summary = pd.DataFrame(
        {
            "class": counts.index,
            "count": counts.values,
            "pct": (counts.values / total * 100).round(4),
        }
    )

    if len(counts) >= 2:
        majority = int(counts.max())
        minority = int(counts.min())
        summary.attrs["imbalance_ratio"] = round(majority / minority, 4) if minority else None

    return summary

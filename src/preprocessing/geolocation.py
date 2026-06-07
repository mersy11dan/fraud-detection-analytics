"""Geolocation enrichment for fraud data using IP range lookups."""

from __future__ import annotations

import logging
from ipaddress import IPv4Address
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FRAUD_DATA_GEOLOCATED_FILENAME,
)
from src.preprocessing.datasets import preprocess_fraud_data, preprocess_ip_country_data
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

UNKNOWN_COUNTRY = "unknown"


def convert_ip_to_integer(
    ip_value: Any,
    *,
    logger: logging.Logger | None = None,
) -> int | None:
    """
    Convert an IP value to a 32-bit integer.

    Fraud_Data stores IPv4 addresses as numeric values (sometimes floats).
    This helper normalizes those values to integers for range-based joins.
    Dotted-decimal strings (e.g. ``192.168.0.1``) are also supported.
    """
    log = logger or get_logger(__name__)

    if ip_value is None or (isinstance(ip_value, float) and np.isnan(ip_value)):
        return None

    if isinstance(ip_value, str):
        stripped = ip_value.strip()
        if not stripped:
            return None
        try:
            return int(IPv4Address(stripped))
        except ValueError:
            log.warning("Could not parse IP string: %s", stripped)
            return None

    try:
        numeric = float(ip_value)
    except (TypeError, ValueError):
        log.warning("Could not parse IP value: %s", ip_value)
        return None

    if numeric < 0 or numeric > 2**32 - 1:
        log.warning("IP value out of IPv4 range: %s", ip_value)
        return None

    return int(np.floor(numeric))


def add_ip_integer_column(
    df: pd.DataFrame,
    *,
    ip_column: str = "ip_address",
    output_column: str = "ip_address_int",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Add an integer IPv4 column derived from a numeric or string IP column."""
    if ip_column not in df.columns:
        raise ValueError(f"IP column '{ip_column}' not found in dataframe.")

    enriched = df.copy()
    enriched[output_column] = enriched[ip_column].apply(
        lambda value: convert_ip_to_integer(value, logger=logger)
    )
    return enriched


def merge_ip_country_ranges(
    fraud_df: pd.DataFrame,
    ip_country_df: pd.DataFrame,
    *,
    ip_integer_column: str = "ip_address_int",
    country_column: str = "country",
    unknown_country: str = UNKNOWN_COUNTRY,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Attach a country label to each transaction using range-based IP lookup.

    Merge strategy
    --------------
    IpAddress_to_Country.csv defines contiguous IPv4 ranges:
        lower_bound_ip_address <= ip <= upper_bound_ip_address

    Steps:
    1. Sort IP ranges by ``lower_bound_ip_address``.
    2. Sort fraud transactions by ``ip_address_int``.
    3. Use ``pd.merge_asof`` (direction='backward') to find the latest range
       whose lower bound is <= the transaction IP.
    4. Keep matches only when the IP also falls below the range upper bound.
    5. Mark non-matching or missing IPs as ``unknown_country``.

    ``merge_asof`` is used because it efficiently performs this interval join
    on sorted data without a costly cross join.
    """
    log = logger or get_logger(__name__)

    required_cols = {
        "lower_bound_ip_address",
        "upper_bound_ip_address",
        "country",
    }
    missing = required_cols - set(ip_country_df.columns)
    if missing:
        raise ValueError(f"IP country dataframe missing columns: {sorted(missing)}")
    if ip_integer_column not in fraud_df.columns:
        raise ValueError(f"Fraud dataframe missing integer IP column: {ip_integer_column}")

    ranges = ip_country_df.copy()
    ranges["lower_bound_ip_address"] = ranges["lower_bound_ip_address"].astype("int64")
    ranges["upper_bound_ip_address"] = ranges["upper_bound_ip_address"].astype("int64")
    ranges = ranges.sort_values("lower_bound_ip_address").reset_index(drop=True)

    transactions = fraud_df.copy()
    transactions["_row_order"] = np.arange(len(transactions))
    sortable = transactions.dropna(subset=[ip_integer_column]).copy()
    sortable["_orig_index"] = sortable.index
    sortable[ip_integer_column] = sortable[ip_integer_column].astype("int64")
    sortable = sortable.sort_values(ip_integer_column)

    # merge_asof resets the index, so _orig_index tracks the source transaction row.
    merged = pd.merge_asof(
        sortable,
        ranges[["lower_bound_ip_address", "upper_bound_ip_address", "country"]],
        left_on=ip_integer_column,
        right_on="lower_bound_ip_address",
        direction="backward",
    )

    # Validate the candidate range: IP must not exceed the range upper bound.
    in_range = merged[ip_integer_column] <= merged["upper_bound_ip_address"]
    lookup_country = merged["country"].where(in_range, pd.NA)

    matched = int(in_range.sum())
    total = len(merged)
    log.info(
        "Range lookup matched %s of %s IPs with valid integers (%.2f%%)",
        matched,
        total,
        (matched / total * 100) if total else 0.0,
    )

    merged[country_column] = lookup_country.fillna(unknown_country).astype("string")
    drop_columns = ["lower_bound_ip_address", "upper_bound_ip_address"]
    if country_column != "country":
        drop_columns.append("country")
    merged = merged.drop(columns=drop_columns, errors="ignore")

    # Map countries back using the preserved source row index from merge_asof.
    transactions[country_column] = unknown_country
    country_by_row = merged.set_index("_orig_index")[country_column]
    transactions.loc[country_by_row.index, country_column] = country_by_row
    transactions = transactions.sort_values("_row_order").drop(columns="_row_order")

    unmatched = int((transactions[country_column] == unknown_country).sum())
    log.info("Transactions without country match: %s", unmatched)
    return transactions.reset_index(drop=True)


def enrich_fraud_data_with_country(
    fraud_df: pd.DataFrame | None = None,
    ip_country_df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run the full geolocation enrichment pipeline on fraud transactions."""
    log = logger or get_logger(__name__)

    fraud = preprocess_fraud_data(fraud_df, data_dir=data_dir, logger=log)
    ip_country = preprocess_ip_country_data(ip_country_df, data_dir=data_dir, logger=log)

    fraud = add_ip_integer_column(fraud, logger=log)
    enriched = merge_ip_country_ranges(fraud, ip_country, logger=log)

    log.info(
        "Geolocation enrichment complete: %s countries detected",
        enriched["country"].nunique(dropna=True),
    )
    return enriched


def fraud_patterns_by_country(
    df: pd.DataFrame,
    *,
    country_column: str = "country",
    target_column: str = "class",
    min_transactions: int = 100,
) -> pd.DataFrame:
    """Summarize fraud rate and volume by country."""
    if country_column not in df.columns:
        raise ValueError(f"Country column '{country_column}' not found in dataframe.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    summary = (
        df.groupby(country_column, dropna=False)[target_column]
        .agg(transactions="count", fraud_cases="sum")
        .reset_index()
    )
    summary["fraud_rate_pct"] = (
        summary["fraud_cases"] / summary["transactions"] * 100
    ).round(3)
    summary = summary[summary["transactions"] >= min_transactions]
    return summary.sort_values("fraud_rate_pct", ascending=False).reset_index(drop=True)


def save_geolocated_fraud_data(
    df: pd.DataFrame,
    *,
    output_dir: Path = DATA_PROCESSED_DIR,
    filename: str = FRAUD_DATA_GEOLOCATED_FILENAME,
    logger: logging.Logger | None = None,
) -> Path:
    """Persist the geolocation-enriched fraud dataset."""
    log = logger or get_logger(__name__)
    output_path = ensure_dir(output_dir) / filename
    df.to_csv(output_path, index=False)
    log.info("Saved geolocated fraud data to %s", output_path)
    return output_path


def run_geolocation_enrichment_pipeline(
    *,
    data_dir: Path = DATA_RAW_DIR,
    output_dir: Path = DATA_PROCESSED_DIR,
    min_transactions: int = 100,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Execute enrichment end-to-end and return data plus country-level analysis.

    Returns
    -------
    enriched_df : pd.DataFrame
        Fraud data with ``ip_address_int`` and ``country`` columns.
    country_patterns : pd.DataFrame
        Fraud rate summary by country.
    output_path : Path
        File path of the saved enriched dataset.
    """
    log = logger or get_logger(__name__)
    enriched = enrich_fraud_data_with_country(data_dir=data_dir, logger=log)
    patterns = fraud_patterns_by_country(enriched, min_transactions=min_transactions)
    output_path = save_geolocated_fraud_data(enriched, output_dir=output_dir, logger=log)
    return enriched, patterns, output_path

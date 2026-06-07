"""Dataset-specific preprocessing pipelines."""

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW_DIR
from src.data.loader import (
    load_creditcard_data,
    load_fraud_data,
    load_ip_country_data,
)
from src.preprocessing.cleaning import (
    convert_timestamps,
    handle_duplicates,
    handle_missing_values,
)
from src.preprocessing.columns import standardize_column_names
from src.utils.logging_config import get_logger


def preprocess_fraud_data(
    df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean and type-correct the e-commerce fraud dataset."""
    log = logger or get_logger(__name__)
    raw = load_fraud_data(data_dir) if df is None else df.copy()
    log.info("Preprocessing Fraud_Data (%s rows, %s columns)", len(raw), len(raw.columns))

    cleaned = standardize_column_names(raw)
    cleaned = convert_timestamps(
        cleaned,
        columns=["signup_time", "purchase_time"],
        logger=log,
    )
    cleaned = handle_duplicates(cleaned, logger=log)
    cleaned = handle_missing_values(
        cleaned,
        required_columns=["signup_time", "purchase_time", "class"],
        numeric_fill="median",
        categorical_fill="unknown",
        logger=log,
    )

    cleaned["purchase_value"] = pd.to_numeric(cleaned["purchase_value"], errors="coerce")
    cleaned["age"] = pd.to_numeric(cleaned["age"], errors="coerce").astype("Int64")
    cleaned["ip_address"] = pd.to_numeric(cleaned["ip_address"], errors="coerce")
    cleaned["class"] = pd.to_numeric(cleaned["class"], errors="coerce").astype("Int64")

    for column in ["source", "browser", "sex", "device_id"]:
        cleaned[column] = cleaned[column].astype("string")

    log.info("Fraud_Data preprocessing complete (%s rows)", len(cleaned))
    return cleaned.reset_index(drop=True)


def preprocess_creditcard_data(
    df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean and type-correct the credit card fraud dataset."""
    log = logger or get_logger(__name__)
    raw = load_creditcard_data(data_dir) if df is None else df.copy()
    log.info("Preprocessing creditcard data (%s rows, %s columns)", len(raw), len(raw.columns))

    cleaned = standardize_column_names(raw)
    cleaned = handle_duplicates(cleaned, logger=log)
    cleaned = handle_missing_values(cleaned, numeric_fill="median", logger=log)

    cleaned["time"] = pd.to_numeric(cleaned["time"], errors="coerce")
    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
    cleaned["class"] = pd.to_numeric(cleaned["class"], errors="coerce").astype("Int64")

    pca_columns = [col for col in cleaned.columns if col.startswith("v")]
    for column in pca_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    log.info("Creditcard preprocessing complete (%s rows)", len(cleaned))
    return cleaned.reset_index(drop=True)


def preprocess_ip_country_data(
    df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean and type-correct the IP-to-country mapping dataset."""
    log = logger or get_logger(__name__)
    raw = load_ip_country_data(data_dir) if df is None else df.copy()
    log.info("Preprocessing IP country data (%s rows, %s columns)", len(raw), len(raw.columns))

    cleaned = standardize_column_names(raw)
    cleaned = handle_duplicates(
        cleaned,
        subset=["lower_bound_ip_address", "upper_bound_ip_address"],
        logger=log,
    )
    cleaned = handle_missing_values(
        cleaned,
        required_columns=["lower_bound_ip_address", "upper_bound_ip_address", "country"],
        categorical_fill="unknown",
        logger=log,
    )

    cleaned["lower_bound_ip_address"] = pd.to_numeric(
        cleaned["lower_bound_ip_address"], errors="coerce"
    ).astype("Int64")
    cleaned["upper_bound_ip_address"] = pd.to_numeric(
        cleaned["upper_bound_ip_address"], errors="coerce"
    ).astype("Int64")
    cleaned["country"] = cleaned["country"].astype("string").str.strip()

    log.info("IP country preprocessing complete (%s rows)", len(cleaned))
    return cleaned.reset_index(drop=True)

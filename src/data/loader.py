"""Dataset loaders for fraud detection analytics."""

from pathlib import Path

import pandas as pd

from src.config import (
    CREDITCARD_FILENAME,
    DATA_RAW_DIR,
    FRAUD_DATA_FILENAME,
    IP_COUNTRY_FILENAME,
)
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)


def _load_csv(
    filename: str,
    data_dir: Path = DATA_RAW_DIR,
    *,
    parse_dates: list[str] | None = None,
    dtype: dict | None = None,
) -> pd.DataFrame:
    """Load a CSV file safely with logging and optional dtype hints."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    LOGGER.info("Loading dataset: %s", path)
    df = pd.read_csv(
        path,
        parse_dates=parse_dates,
        dtype=dtype,
        low_memory=False,
    )
    LOGGER.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), filename)
    return df


def load_creditcard_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    return _load_csv(CREDITCARD_FILENAME, data_dir)


def load_fraud_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the e-commerce fraud dataset."""
    return _load_csv(
        FRAUD_DATA_FILENAME,
        data_dir,
        dtype={
            "user_id": "Int64",
            "purchase_value": "float64",
            "age": "Int64",
            "class": "Int64",
        },
    )


def load_ip_country_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the IP address to country mapping dataset."""
    return _load_csv(
        IP_COUNTRY_FILENAME,
        data_dir,
        dtype={
            "lower_bound_ip_address": "Int64",
            "upper_bound_ip_address": "Int64",
            "country": "string",
        },
    )

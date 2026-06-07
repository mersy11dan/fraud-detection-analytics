"""Dataset loaders for fraud detection analytics."""

from pathlib import Path

import pandas as pd

from src.config import (
    CREDITCARD_FILENAME,
    DATA_RAW_DIR,
    FRAUD_DATA_FILENAME,
    IP_COUNTRY_FILENAME,
)


def _load_csv(filename: str, data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load a CSV file from the raw data directory."""
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def load_creditcard_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the credit card fraud dataset."""
    return _load_csv(CREDITCARD_FILENAME, data_dir)


def load_fraud_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the e-commerce fraud dataset."""
    return _load_csv(FRAUD_DATA_FILENAME, data_dir)


def load_ip_country_data(data_dir: Path = DATA_RAW_DIR) -> pd.DataFrame:
    """Load the IP address to country mapping dataset."""
    return _load_csv(IP_COUNTRY_FILENAME, data_dir)

"""Integration tests for dataset preprocessing pipelines."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_RAW_DIR
from src.preprocessing.datasets import (
    preprocess_creditcard_data,
    preprocess_fraud_data,
    preprocess_ip_country_data,
)
from src.preprocessing.inspect import class_imbalance_summary, missing_values_summary


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "Fraud_Data.csv").exists(),
    reason="Fraud_Data.csv not available",
)
def test_preprocess_fraud_data() -> None:
    df = preprocess_fraud_data()

    assert "signup_time" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])
    assert pd.api.types.is_datetime64_any_dtype(df["purchase_time"])
    assert df["class"].notna().all()
    assert missing_values_summary(df)["missing_count"].sum() == 0


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "creditcard.csv").exists(),
    reason="creditcard.csv not available",
)
def test_preprocess_creditcard_data() -> None:
    df = preprocess_creditcard_data()

    assert "amount" in df.columns
    assert "class" in df.columns
    assert df["class"].dtype.name == "Int64"
    imbalance = class_imbalance_summary(df, target_column="class")
    assert len(imbalance) == 2


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "IpAddress_to_Country.csv").exists(),
    reason="IpAddress_to_Country.csv not available",
)
def test_preprocess_ip_country_data() -> None:
    df = preprocess_ip_country_data()

    assert df["country"].notna().all()
    assert df["lower_bound_ip_address"].dtype.name == "Int64"
    assert df["upper_bound_ip_address"].dtype.name == "Int64"

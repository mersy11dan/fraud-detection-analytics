"""Tests for geolocation enrichment utilities."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_RAW_DIR
from src.preprocessing.geolocation import (
    add_ip_integer_column,
    convert_ip_to_integer,
    enrich_fraud_data_with_country,
    fraud_patterns_by_country,
    merge_ip_country_ranges,
)


def test_convert_ip_to_integer_from_float() -> None:
    assert convert_ip_to_integer(732758368.79972) == 732758368


def test_convert_ip_to_integer_from_string() -> None:
    assert convert_ip_to_integer("192.168.0.1") == 3232235521


def test_convert_ip_to_integer_invalid() -> None:
    assert convert_ip_to_integer(None) is None
    assert convert_ip_to_integer("not-an-ip") is None


def test_merge_ip_country_ranges_assigns_country() -> None:
    fraud_df = pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "ip_address_int": [16777216, 16777472, 999],
            "class": [0, 1, 0],
        }
    )
    ip_country_df = pd.DataFrame(
        {
            "lower_bound_ip_address": [16777216, 16777472],
            "upper_bound_ip_address": [16777471, 16777727],
            "country": ["Australia", "China"],
        }
    )

    enriched = merge_ip_country_ranges(fraud_df, ip_country_df)

    assert enriched.loc[0, "country"] == "Australia"
    assert enriched.loc[1, "country"] == "China"
    assert enriched.loc[2, "country"] == "unknown"


def test_add_ip_integer_column() -> None:
    df = pd.DataFrame({"ip_address": [16777216.2, 16777472.9]})
    result = add_ip_integer_column(df)

    assert result["ip_address_int"].tolist() == [16777216, 16777472]


def test_fraud_patterns_by_country() -> None:
    df = pd.DataFrame(
        {
            "country": ["A", "A", "A", "B", "B"],
            "class": [0, 0, 1, 1, 1],
        }
    )
    summary = fraud_patterns_by_country(df, min_transactions=2)

    assert summary.loc[summary["country"] == "B", "fraud_rate_pct"].iloc[0] == 100.0
    assert summary.loc[summary["country"] == "A", "fraud_rate_pct"].iloc[0] == pytest.approx(
        33.333, rel=1e-2
    )


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "Fraud_Data.csv").exists()
    or not (DATA_RAW_DIR / "IpAddress_to_Country.csv").exists(),
    reason="Required raw datasets not available",
)
def test_enrich_fraud_data_with_country_integration() -> None:
    enriched = enrich_fraud_data_with_country()

    assert "country" in enriched.columns
    assert "ip_address_int" in enriched.columns
    assert enriched["country"].notna().all()
    assert len(enriched) > 0

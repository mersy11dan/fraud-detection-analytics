"""Focused tests for preprocessing and feature engineering helpers."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.features.temporal import add_time_since_signup
from src.modeling.imbalance import compare_class_distributions
from src.preprocessing.cleaning import handle_duplicates, handle_missing_values
from src.preprocessing.geolocation import enrich_fraud_data_with_country
from src.preprocessing.inspect import class_imbalance_summary, duplicate_check, missing_values_summary


def _dirty_fraud_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3],
            "signup_time": [
                "2015-01-01 10:00:00",
                "2015-01-01 10:00:00",
                "2015-01-02 09:00:00",
                None,
            ],
            "purchase_time": [
                "2015-01-01 11:00:00",
                "2015-01-01 11:00:00",
                None,
                "2015-01-03 12:00:00",
            ],
            "purchase_value": [25.0, 25.0, 40.0, 15.0],
            "device_id": ["D1", "D1", "D2", "D3"],
            "source": ["SEO", "SEO", "Ads", "Direct"],
            "browser": ["Chrome", "Chrome", "Safari", "Firefox"],
            "sex": ["M", "M", "F", "M"],
            "age": [30, 30, 28, 35],
            "ip_address": [16777216.5, 16777216.5, 16777472.1, 999.0],
            "class": [0, 0, 1, 0],
        }
    )


def _ip_country_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lower_bound_ip_address": [16777216, 16777472],
            "upper_bound_ip_address": [16777471, 16777727],
            "country": ["Australia", "China"],
        }
    )


def _fraud_frame_for_geolocation() -> pd.DataFrame:
    """Valid fraud rows for geolocation enrichment tests."""
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "signup_time": [
                "2015-01-01 10:00:00",
                "2015-01-02 09:00:00",
                "2015-01-03 08:00:00",
            ],
            "purchase_time": [
                "2015-01-01 11:00:00",
                "2015-01-02 10:00:00",
                "2015-01-03 09:00:00",
            ],
            "purchase_value": [25.0, 40.0, 15.0],
            "device_id": ["D1", "D2", "D3"],
            "source": ["SEO", "Ads", "Direct"],
            "browser": ["Chrome", "Safari", "Firefox"],
            "sex": ["M", "F", "M"],
            "age": [30, 28, 35],
            "ip_address": [16777216.5, 16777472.1, 999.0],
            "class": [0, 1, 0],
        }
    )


class TestMissingValuesAndDuplicates:
    """Requirement: missing values and duplicates are handled correctly."""

    def test_handle_missing_values_drops_required_rows_and_fills_others(self) -> None:
        df = _dirty_fraud_frame()
        cleaned = handle_missing_values(
            df,
            required_columns=["signup_time", "purchase_time"],
            numeric_fill="median",
            categorical_fill="unknown",
        )

        assert len(cleaned) == 2
        assert cleaned["signup_time"].notna().all()
        assert cleaned["purchase_time"].notna().all()
        assert cleaned.isna().sum().sum() == 0

    def test_missing_values_summary_counts_nulls(self) -> None:
        df = _dirty_fraud_frame()
        summary = missing_values_summary(df)

        assert summary.loc["signup_time", "missing_count"] == 1
        assert summary.loc["purchase_time", "missing_count"] == 1

    def test_handle_duplicates_removes_duplicate_rows(self) -> None:
        df = _dirty_fraud_frame()
        cleaned = handle_duplicates(df)

        assert len(cleaned) == 3

    def test_duplicate_check_reports_duplicate_rows(self) -> None:
        df = _dirty_fraud_frame()
        report = duplicate_check(df)

        assert report["duplicate_rows_to_remove"] == 1
        assert report["rows_in_duplicate_groups"] == 2


class TestTimeSinceSignup:
    """Requirement: time_since_signup is created correctly."""

    def test_add_time_since_signup_hours(self) -> None:
        base = datetime(2015, 1, 1, 10, 0, 0)
        df = pd.DataFrame(
            {
                "signup_time": [base, base + timedelta(days=1)],
                "purchase_time": [base + timedelta(hours=2), base + timedelta(days=1, hours=6)],
            }
        )

        result = add_time_since_signup(df)

        assert "time_since_signup_hours" in result.columns
        assert result["time_since_signup_hours"].tolist() == [2.0, 6.0]

    def test_time_since_signup_zero_for_instant_purchase(self) -> None:
        timestamp = datetime(2015, 6, 1, 12, 0, 0)
        df = pd.DataFrame(
            {
                "signup_time": [timestamp],
                "purchase_time": [timestamp],
            }
        )

        result = add_time_since_signup(df)

        assert result["time_since_signup_hours"].iloc[0] == 0.0


class TestIpCountryEnrichment:
    """Requirement: IP-to-country enrichment returns expected columns."""

    def test_enrich_fraud_data_with_country_adds_expected_columns(self) -> None:
        enriched = enrich_fraud_data_with_country(
            fraud_df=_fraud_frame_for_geolocation(),
            ip_country_df=_ip_country_frame(),
        )

        expected_columns = {
            "user_id",
            "signup_time",
            "purchase_time",
            "purchase_value",
            "device_id",
            "source",
            "browser",
            "sex",
            "age",
            "ip_address",
            "class",
            "ip_address_int",
            "country",
        }
        assert expected_columns.issubset(set(enriched.columns))

    def test_enrich_fraud_data_with_country_assigns_country_labels(self) -> None:
        enriched = enrich_fraud_data_with_country(
            fraud_df=_fraud_frame_for_geolocation(),
            ip_country_df=_ip_country_frame(),
        )

        assert enriched.loc[0, "country"] == "Australia"
        assert enriched.loc[1, "country"] == "China"
        assert enriched.loc[2, "country"] == "unknown"
        assert enriched["country"].notna().all()
        assert enriched["ip_address_int"].dtype.name in {"int64", "Int64"}


class TestImbalanceSummary:
    """Requirement: imbalance summary logic works on a small synthetic dataset."""

    def test_class_imbalance_summary_on_synthetic_data(self) -> None:
        df = pd.DataFrame({"class": [0, 0, 0, 0, 0, 1, 1]})
        summary = class_imbalance_summary(df, target_column="class")

        assert summary["count"].tolist() == [5, 2]
        assert summary["pct"].tolist() == pytest.approx([71.4286, 28.5714], rel=1e-3)
        assert summary.attrs["imbalance_ratio"] == pytest.approx(2.5)

    def test_compare_class_distributions_before_and_after_resampling(self) -> None:
        y_train_before = pd.Series([0] * 8 + [1] * 2, name="class")
        y_train_after = pd.Series([0] * 5 + [1] * 5, name="class")
        y_test = pd.Series([0] * 4 + [1] * 1, name="class")

        comparison = compare_class_distributions(y_train_before, y_train_after, y_test)

        before = comparison[comparison["stage"] == "train_before_resampling"]
        after = comparison[comparison["stage"] == "train_after_resampling"]
        test = comparison[comparison["stage"] == "test_holdout_unmodified"]

        assert before.loc[before["class"] == 0, "pct"].iloc[0] == 80.0
        assert after.loc[after["class"] == 0, "pct"].iloc[0] == 50.0
        assert test["count"].sum() == 5
        assert set(comparison["stage"]) == {
            "train_before_resampling",
            "train_after_resampling",
            "test_holdout_unmodified",
        }

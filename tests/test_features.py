"""Tests for fraud feature engineering."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.config import DATA_RAW_DIR
from src.features.pipeline import FraudFeaturePipeline, engineer_fraud_features
from src.features.temporal import add_purchase_time_features, add_time_since_signup
from src.features.velocity import add_user_velocity_features


def _sample_fraud_df() -> pd.DataFrame:
    base = datetime(2015, 1, 1, 10, 0, 0)
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "signup_time": [base, base, base + timedelta(days=1)],
            "purchase_time": [
                base + timedelta(hours=1),
                base + timedelta(hours=3),
                base + timedelta(days=1, hours=2),
            ],
            "purchase_value": [20.0, 35.0, 15.0],
            "device_id": ["A", "A", "B"],
            "source": ["SEO", "Ads", "SEO"],
            "browser": ["Chrome", "Chrome", "Safari"],
            "sex": ["M", "M", "F"],
            "age": [30, 30, 40],
            "ip_address": [1000.0, 1001.0, 2000.0],
            "class": [0, 1, 0],
        }
    )


def test_add_time_since_signup() -> None:
    df = _sample_fraud_df()
    result = add_time_since_signup(df)

    assert result["time_since_signup_hours"].tolist() == [1.0, 3.0, 2.0]


def test_add_purchase_time_features() -> None:
    df = _sample_fraud_df()
    result = add_purchase_time_features(df)

    assert result["hour_of_day"].tolist() == [11, 13, 12]
    assert result["day_of_week"].tolist() == [3, 3, 4]


def test_add_user_velocity_features() -> None:
    df = _sample_fraud_df()
    result = add_user_velocity_features(df)

    assert "txn_count_last_1h" in result.columns
    assert "txn_count_last_24h" in result.columns
    assert "user_cumulative_txn_count" in result.columns
    assert result.loc[result["user_id"] == 1, "user_cumulative_txn_count"].tolist() == [1, 2]


def test_fraud_feature_pipeline_fit_transform() -> None:
    df = _sample_fraud_df()
    pipeline = FraudFeaturePipeline(include_country=False)
    features, target = pipeline.fit_transform(df)

    assert len(features) == len(df)
    assert len(target) == len(df)
    assert features.shape[1] > 5
    assert set(target.tolist()) == {0, 1}


def test_engineer_fraud_features_adds_expected_columns() -> None:
    df = _sample_fraud_df()
    engineered = engineer_fraud_features(df=df)

    expected = {
        "time_since_signup_hours",
        "hour_of_day",
        "day_of_week",
        "txn_count_last_1h",
        "txn_count_last_24h",
        "txn_count_last_168h",
        "hours_since_last_txn",
        "user_cumulative_txn_count",
        "user_txn_velocity_per_day",
    }
    assert expected.issubset(engineered.columns)


@pytest.mark.skipif(
    not (DATA_RAW_DIR / "Fraud_Data.csv").exists(),
    reason="Fraud_Data.csv not available",
)
def test_engineer_fraud_features_integration() -> None:
    engineered = engineer_fraud_features(include_country=True)

    assert len(engineered) > 0
    assert "time_since_signup_hours" in engineered.columns
    assert engineered["time_since_signup_hours"].notna().all()

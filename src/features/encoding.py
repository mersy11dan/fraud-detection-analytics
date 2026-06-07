"""Scaling and categorical encoding for fraud feature matrices."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FraudFeatureConfig:
    """Configuration for fraud feature matrix construction."""

    numeric_features: list[str] = field(
        default_factory=lambda: [
            "purchase_value",
            "age",
            "time_since_signup_hours",
            "hour_of_day",
            "day_of_week",
            "txn_count_last_1h",
            "txn_count_last_24h",
            "txn_count_last_168h",
            "hours_since_last_txn",
            "user_cumulative_txn_count",
            "user_txn_velocity_per_day",
        ]
    )
    categorical_features: list[str] = field(
        default_factory=lambda: ["source", "browser", "sex"]
    )
    optional_categorical_features: list[str] = field(
        default_factory=lambda: ["country"]
    )
    target_column: str = "class"
    drop_columns: list[str] = field(
        default_factory=lambda: [
            "user_id",
            "signup_time",
            "purchase_time",
            "device_id",
            "ip_address",
            "ip_address_int",
        ]
    )


def build_preprocessor(
    config: FraudFeatureConfig,
    *,
    available_columns: list[str],
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Build a sklearn ColumnTransformer for scaling and one-hot encoding.

    High-cardinality identifiers such as ``device_id`` are excluded from
    one-hot encoding to avoid sparse, unstable feature spaces.
    """
    numeric_features = [col for col in config.numeric_features if col in available_columns]
    categorical_features = [
        col
        for col in config.categorical_features + config.optional_categorical_features
        if col in available_columns
    ]

    if not numeric_features:
        raise ValueError("No numeric features available for scaling.")
    if not categorical_features:
        raise ValueError("No categorical features available for encoding.")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_features, categorical_features


def transform_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    *,
    numeric_features: list[str],
    categorical_features: list[str],
    target_column: str = "class",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Apply a fitted preprocessor and return a model-ready feature matrix."""
    feature_columns = numeric_features + categorical_features
    matrix = preprocessor.transform(df[feature_columns])

    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(matrix.shape[1])]

    features = pd.DataFrame(matrix, columns=feature_names, index=df.index)

    if target_column in df.columns:
        target = df[target_column].astype("int64")
        return features, target

    return features, None

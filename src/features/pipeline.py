"""End-to-end feature engineering pipeline for Fraud_Data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FRAUD_DATA_ENGINEERED_FILENAME,
    FRAUD_DATA_FEATURES_FILENAME,
)
from src.features.encoding import (
    FraudFeatureConfig,
    build_preprocessor,
    transform_features,
)
from src.features.temporal import add_purchase_time_features, add_time_since_signup
from src.features.velocity import add_user_velocity_features
from src.preprocessing.datasets import preprocess_fraud_data
from src.preprocessing.geolocation import enrich_fraud_data_with_country
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir

class FraudFeaturePipeline:
    """
    Reusable feature engineering pipeline for Fraud_Data.

    Steps
    -----
    1. Clean base transactions (optionally with country enrichment).
    2. Engineer temporal and user velocity features.
    3. Scale numeric columns and one-hot encode categorical columns.
    4. Persist engineered tables to ``data/processed/``.
    """

    def __init__(
        self,
        *,
        config: FraudFeatureConfig | None = None,
        include_country: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or FraudFeatureConfig()
        self.include_country = include_country
        self.logger = logger or get_logger(__name__)
        self.preprocessor = None
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal and velocity features to a cleaned fraud dataframe."""
        self.logger.info("Creating temporal features")
        engineered = add_time_since_signup(df)
        engineered = add_purchase_time_features(engineered)

        self.logger.info("Creating user velocity features")
        engineered = add_user_velocity_features(engineered)
        self.logger.info("Derived features complete (%s columns)", len(engineered.columns))
        return engineered

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        engineered: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Fit preprocessing transformers and return model-ready features."""
        feature_table = engineered if engineered is not None else self.create_derived_features(df)

        self.preprocessor, self.numeric_features, self.categorical_features = (
            build_preprocessor(self.config, available_columns=feature_table.columns.tolist())
        )

        self.logger.info(
            "Fitting preprocessor on %s numeric and %s categorical features",
            len(self.numeric_features),
            len(self.categorical_features),
        )
        self.preprocessor.fit(feature_table)

        features, target = transform_features(
            feature_table,
            self.preprocessor,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            target_column=self.config.target_column,
        )
        if target is None:
            raise ValueError(f"Target column '{self.config.target_column}' not found.")

        return features, target

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
        """Transform new data with a previously fitted preprocessor."""
        if self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        engineered = self.create_derived_features(df)
        return transform_features(
            engineered,
            self.preprocessor,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            target_column=self.config.target_column,
        )


def _load_base_fraud_data(
    df: pd.DataFrame | None,
    *,
    data_dir: Path,
    include_country: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if df is not None:
        return preprocess_fraud_data(df, logger=logger)

    if include_country:
        try:
            return enrich_fraud_data_with_country(data_dir=data_dir, logger=logger)
        except FileNotFoundError:
            logger.warning("IP country mapping unavailable; continuing without country feature.")

    return preprocess_fraud_data(data_dir=data_dir, logger=logger)


def engineer_fraud_features(
    df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    include_country: bool = True,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Return fraud data with engineered temporal and velocity features."""
    log = logger or get_logger(__name__)
    base = _load_base_fraud_data(df, data_dir=data_dir, include_country=include_country, logger=log)
    pipeline = FraudFeaturePipeline(include_country=include_country, logger=log)
    return pipeline.create_derived_features(base)


def build_fraud_feature_matrix(
    df: pd.DataFrame | None = None,
    *,
    data_dir: Path = DATA_RAW_DIR,
    include_country: bool = True,
    config: FraudFeatureConfig | None = None,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build a scaled and encoded fraud feature matrix.

    Returns
    -------
    features : pd.DataFrame
        Model-ready design matrix.
    target : pd.Series
        Fraud labels.
    engineered : pd.DataFrame
        Intermediate table with raw engineered columns before encoding.
    """
    log = logger or get_logger(__name__)
    base = _load_base_fraud_data(df, data_dir=data_dir, include_country=include_country, logger=log)

    pipeline = FraudFeaturePipeline(
        config=config,
        include_country=include_country,
        logger=log,
    )
    engineered = pipeline.create_derived_features(base)
    features, target = pipeline.fit_transform(base, engineered=engineered)
    return features, target, engineered


def save_engineered_fraud_data(
    *,
    data_dir: Path = DATA_RAW_DIR,
    output_dir: Path = DATA_PROCESSED_DIR,
    include_country: bool = True,
    logger: logging.Logger | None = None,
) -> dict[str, Path]:
    """Run the full pipeline and save engineered and model-ready outputs."""
    log = logger or get_logger(__name__)
    features, target, engineered = build_fraud_feature_matrix(
        data_dir=data_dir,
        include_country=include_country,
        logger=log,
    )

    output_path = ensure_dir(output_dir)
    engineered_path = output_path / FRAUD_DATA_ENGINEERED_FILENAME
    features_path = output_path / FRAUD_DATA_FEATURES_FILENAME

    engineered.to_csv(engineered_path, index=False)

    feature_table = features.copy()
    feature_table["class"] = target.values
    feature_table.to_csv(features_path, index=False)

    log.info("Saved engineered features to %s", engineered_path)
    log.info("Saved model-ready feature matrix to %s", features_path)
    log.info("Feature matrix shape: %s", feature_table.shape)

    return {
        "engineered": engineered_path,
        "features": features_path,
    }

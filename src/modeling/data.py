"""Data loading and splitting utilities for fraud classification modeling."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FRAUD_DATA_FEATURES_FILENAME,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features.pipeline import build_fraud_feature_matrix
from src.utils.logging_config import get_logger

TARGET_COLUMN = "class"


@dataclass
class ModelingSplit:
    """Stratified train/test split for classification modeling."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]


def load_fraud_feature_matrix(
    *,
    data_dir: Path = DATA_PROCESSED_DIR,
    raw_data_dir: Path = DATA_RAW_DIR,
    features_filename: str = FRAUD_DATA_FEATURES_FILENAME,
    target_column: str = TARGET_COLUMN,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed Fraud_Data feature matrix.

    Reads ``data/processed/fraud_data_features.csv`` when available; otherwise
    builds the matrix from raw data via the feature engineering pipeline.
    """
    log = logger or get_logger(__name__)
    features_path = data_dir / features_filename

    if features_path.exists():
        log.info("Loading processed feature matrix from %s", features_path)
        feature_table = pd.read_csv(features_path)
    else:
        log.info("Processed feature matrix not found; building from raw data")
        features, target, _ = build_fraud_feature_matrix(data_dir=raw_data_dir, logger=log)
        return features, target

    if target_column not in feature_table.columns:
        raise ValueError(f"Target column '{target_column}' not found in {features_path}")

    target = feature_table.pop(target_column)
    target.name = target_column
    return feature_table, target


def stratified_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ModelingSplit:
    """Split features and target with stratification on the fraud label."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    return ModelingSplit(
        x_train=x_train.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=features.columns.tolist(),
    )


def prepare_fraud_modeling_data(
    *,
    data_dir: Path = DATA_PROCESSED_DIR,
    raw_data_dir: Path = DATA_RAW_DIR,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> ModelingSplit:
    """Load processed Fraud_Data features and return a stratified train/test split."""
    log = logger or get_logger(__name__)
    features, target = load_fraud_feature_matrix(
        data_dir=data_dir,
        raw_data_dir=raw_data_dir,
        logger=log,
    )
    split = stratified_train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
    log.info(
        "Prepared modeling split: train=%s, test=%s, features=%s",
        len(split.x_train),
        len(split.x_test),
        len(split.feature_names),
    )
    return split

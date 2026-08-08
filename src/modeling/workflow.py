"""End-to-end fraud modeling workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.config import (
    CREDITCARD_CONFUSION_DIR,
    CREDITCARD_OUTPUTS_DIR,
    CREDITCARD_PLOTS_DIR,
    MODELING_CONFUSION_DIR,
    MODELING_OUTPUTS_DIR,
    MODELING_PLOTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.modeling.data import (
    load_creditcard_feature_matrix,
    load_fraud_feature_matrix,
    stratified_train_test_split,
)
from src.modeling.reporting import ModelingReportPaths, save_modeling_report
from src.modeling.training import (
    ModelTrainingResult,
    compare_model_results,
    identify_best_model,
    train_classifier,
)
from src.modeling.tuning import TuningResult, get_tuning_configs, tune_classifier
from src.utils.logging_config import get_logger

DEFAULT_MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]


@dataclass
class ModelingWorkflowResult:
    """Container for a full fraud modeling run."""

    dataset_name: str
    results: list[ModelTrainingResult]
    comparison: pd.DataFrame
    best_result: ModelTrainingResult
    tuning_results: list[TuningResult]
    report_paths: ModelingReportPaths | None = None


def _smote_k_neighbors(minority_count: int) -> int:
    """Choose a safe SMOTE k_neighbors for highly imbalanced training splits."""
    return max(1, min(5, minority_count - 1))


def _subsample_for_tuning(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    max_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Stratified subsample for hyperparameter search on large training sets."""
    if len(x_train) <= max_rows:
        return x_train, y_train
    x_sub, _, y_sub, _ = train_test_split(
        x_train,
        y_train,
        train_size=max_rows,
        stratify=y_train,
        random_state=random_state,
    )
    return x_sub.reset_index(drop=True), y_sub.reset_index(drop=True)


def _run_modeling_workflow(
    *,
    dataset_name: str,
    load_features: Callable[..., tuple[pd.DataFrame, pd.Series]],
    output_dir: Path = MODELING_OUTPUTS_DIR,
    plots_dir: Path | None = None,
    confusion_dir: Path | None = None,
    model_names: list[str] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv: int = 3,
    save_report: bool = True,
    store_training_data: bool = False,
    logger: logging.Logger | None = None,
) -> ModelingWorkflowResult:
    """Shared modeling workflow for Fraud_Data and creditcard.csv."""
    log = logger or get_logger(__name__)
    models = model_names or DEFAULT_MODEL_ORDER

    features, target = load_features(logger=log)
    split = stratified_train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    minority_count = int(split.y_train.sum())
    smote_kwargs: dict = {
        "random_state": random_state,
        "k_neighbors": _smote_k_neighbors(minority_count),
    }
    if dataset_name == "creditcard":
        # Partial oversampling — full 50/50 SMOTE on 283k rows is costly and over-synthetic.
        smote_kwargs["sampling_strategy"] = 0.05
    smote = SMOTE(**smote_kwargs)
    x_resampled, y_resampled = smote.fit_resample(split.x_train, split.y_train)
    x_train_resampled = pd.DataFrame(x_resampled, columns=split.feature_names)
    y_train_resampled = pd.Series(y_resampled, name="class")

    log.info(
        "%s workflow split: train=%s, test=%s, resampled_train=%s, minority_train=%s",
        dataset_name,
        len(split.x_train),
        len(split.x_test),
        len(x_train_resampled),
        minority_count,
    )

    tuning_configs = get_tuning_configs(random_state=random_state)
    tuning_rows = 50_000 if dataset_name == "creditcard" else len(split.x_train)
    x_tune, y_tune = _subsample_for_tuning(
        split.x_train,
        split.y_train,
        max_rows=tuning_rows,
        random_state=random_state,
    )
    if len(x_tune) < len(split.x_train):
        log.info(
            "%s tuning subsample: %s rows (from %s)",
            dataset_name,
            len(x_tune),
            len(split.x_train),
        )

    tuning_results: list[TuningResult] = []
    model_results: list[ModelTrainingResult] = []

    for model_name in models:
        tuning = tune_classifier(
            model_name,
            x_tune,
            y_tune,
            config=tuning_configs[model_name],
            cv=cv,
            random_state=random_state,
            logger=log,
        )
        tuning_results.append(tuning)

        result = train_classifier(
            tuning.best_estimator,
            x_train_resampled,
            y_train_resampled,
            split.x_test,
            split.y_test,
            model_name=model_name,
            store_training_data=store_training_data,
            random_state=random_state,
            logger=log,
        )
        model_results.append(result)

    comparison = compare_model_results(model_results, sort_by="auc_pr")
    best_result = identify_best_model(model_results, metric="auc_pr")

    report_paths = None
    if save_report:
        report_paths = save_modeling_report(
            model_results,
            output_dir=output_dir,
            plots_dir=plots_dir or output_dir / "plots",
            confusion_dir=confusion_dir or output_dir / "confusion_matrices",
            dataset_label=dataset_name,
            logger=log,
        )

    log.info(
        "%s best model: %s (PR-AUC=%.4f, F1=%.4f)",
        dataset_name,
        best_result.model_name,
        best_result.metrics.auc_pr,
        best_result.metrics.f1,
    )

    return ModelingWorkflowResult(
        dataset_name=dataset_name,
        results=model_results,
        comparison=comparison,
        best_result=best_result,
        tuning_results=tuning_results,
        report_paths=report_paths,
    )


def run_fraud_modeling_workflow(
    *,
    model_names: list[str] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv: int = 3,
    save_report: bool = True,
    store_training_data: bool = False,
    logger: logging.Logger | None = None,
) -> ModelingWorkflowResult:
    """Run the full Fraud_Data modeling workflow."""
    return _run_modeling_workflow(
        dataset_name="Fraud_Data",
        load_features=load_fraud_feature_matrix,
        output_dir=MODELING_OUTPUTS_DIR,
        plots_dir=MODELING_PLOTS_DIR,
        confusion_dir=MODELING_CONFUSION_DIR,
        model_names=model_names,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        save_report=save_report,
        store_training_data=store_training_data,
        logger=logger,
    )


def run_creditcard_modeling_workflow(
    *,
    model_names: list[str] | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    cv: int = 3,
    save_report: bool = True,
    store_training_data: bool = False,
    logger: logging.Logger | None = None,
) -> ModelingWorkflowResult:
    """Run the full creditcard.csv modeling workflow."""
    return _run_modeling_workflow(
        dataset_name="creditcard",
        load_features=load_creditcard_feature_matrix,
        output_dir=CREDITCARD_OUTPUTS_DIR,
        plots_dir=CREDITCARD_PLOTS_DIR,
        confusion_dir=CREDITCARD_CONFUSION_DIR,
        model_names=model_names,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        save_report=save_report,
        store_training_data=store_training_data,
        logger=logger,
    )

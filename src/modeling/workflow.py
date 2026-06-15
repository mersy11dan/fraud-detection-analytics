"""End-to-end fraud modeling workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_STATE, TEST_SIZE
from src.modeling.data import load_fraud_feature_matrix, stratified_train_test_split
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

    results: list[ModelTrainingResult]
    comparison: pd.DataFrame
    best_result: ModelTrainingResult
    tuning_results: list[TuningResult]
    report_paths: ModelingReportPaths | None = None


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
    """
    Run the full Fraud_Data modeling workflow.

    Steps
    -----
    1. Load processed features.
    2. Stratified train/test split.
    3. Tune each model on the pre-SMOTE training split (CV, AUC-PR).
    4. Refit tuned models on SMOTE-balanced training data.
    5. Evaluate on the untouched holdout set.
    6. Optionally save report-ready outputs to ``reports/outputs/``.
    """
    log = logger or get_logger(__name__)
    models = model_names or DEFAULT_MODEL_ORDER

    features, target = load_fraud_feature_matrix(logger=log)
    split = stratified_train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    smote = SMOTE(random_state=random_state)
    x_resampled, y_resampled = smote.fit_resample(split.x_train, split.y_train)
    x_train_resampled = pd.DataFrame(x_resampled, columns=split.feature_names)
    y_train_resampled = pd.Series(y_resampled, name="class")

    log.info(
        "Workflow split: train=%s, test=%s, resampled_train=%s",
        len(split.x_train),
        len(split.x_test),
        len(x_train_resampled),
    )

    tuning_configs = get_tuning_configs(random_state=random_state)
    tuning_results: list[TuningResult] = []
    model_results: list[ModelTrainingResult] = []

    for model_name in models:
        tuning = tune_classifier(
            model_name,
            split.x_train,
            split.y_train,
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
        report_paths = save_modeling_report(model_results, logger=log)

    log.info(
        "Best model: %s (PR-AUC=%.4f, F1=%.4f)",
        best_result.model_name,
        best_result.metrics.auc_pr,
        best_result.metrics.f1,
    )

    return ModelingWorkflowResult(
        results=model_results,
        comparison=comparison,
        best_result=best_result,
        tuning_results=tuning_results,
        report_paths=report_paths,
    )

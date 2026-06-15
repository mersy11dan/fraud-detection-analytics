"""Classifier training and result comparison utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import RANDOM_STATE, REPORTS_DIR
from src.modeling.metrics import ClassificationMetrics, compute_classification_metrics
from src.utils.logging_config import get_logger
from src.utils.paths import ensure_dir


@dataclass
class ModelTrainingResult:
    """
    Container for a fitted classifier and its holdout evaluation.

    The fitted ``estimator`` and ``feature_names`` are kept accessible for
    downstream SHAP analysis in notebooks.
    """

    model_name: str
    estimator: ClassifierMixin
    y_true: pd.Series
    y_pred: np.ndarray
    y_score: np.ndarray
    metrics: ClassificationMetrics
    feature_names: list[str]
    x_train: pd.DataFrame | None = None


def get_default_estimators(
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, ClassifierMixin]:
    """Return baseline sklearn-compatible classifiers for fraud detection."""
    from xgboost import XGBClassifier

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def _positive_class_scores(estimator: ClassifierMixin, features: pd.DataFrame) -> np.ndarray:
    """Extract fraud-class scores from ``predict_proba`` or ``decision_function``."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]

    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(features)

    raise AttributeError(
        f"Estimator {type(estimator).__name__} lacks predict_proba and decision_function."
    )


def train_classifier(
    model: ClassifierMixin | str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    model_name: str | None = None,
    store_training_data: bool = False,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> ModelTrainingResult:
    """
    Fit a classifier and evaluate it on a holdout test set.

    Pass a string model name (``"logistic_regression"``, ``"random_forest"``,
    ``"xgboost"``) or any sklearn-compatible estimator. Set
    ``store_training_data=True`` to retain ``x_train`` for SHAP background
    sampling in notebooks.
    """
    log = logger or get_logger(__name__)

    if isinstance(model, str):
        estimators = get_default_estimators(random_state=random_state)
        if model not in estimators:
            supported = ", ".join(sorted(estimators))
            raise ValueError(f"Unknown model '{model}'. Supported defaults: {supported}")
        resolved_name = model
        estimator = estimators[model]
    else:
        resolved_name = model_name or type(model).__name__
        estimator = model

    log.info("Training %s on %s rows", resolved_name, len(x_train))
    fitted = estimator.fit(x_train, y_train)

    y_pred = fitted.predict(x_test)
    y_score = _positive_class_scores(fitted, x_test)
    metrics = compute_classification_metrics(y_test, y_pred, y_score)

    log.info(
        "%s holdout metrics: precision=%.4f recall=%.4f f1=%.4f auc_pr=%.4f",
        resolved_name,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.auc_pr,
    )

    return ModelTrainingResult(
        model_name=resolved_name,
        estimator=fitted,
        y_true=y_test.reset_index(drop=True),
        y_pred=y_pred,
        y_score=y_score,
        metrics=metrics,
        feature_names=x_train.columns.tolist(),
        x_train=x_train.reset_index(drop=True) if store_training_data else None,
    )


def train_multiple_classifiers(
    models: dict[str, ClassifierMixin] | list[str],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    store_training_data: bool = False,
    random_state: int = RANDOM_STATE,
    logger: logging.Logger | None = None,
) -> list[ModelTrainingResult]:
    """Train and evaluate multiple classifiers on the same split."""
    if isinstance(models, list):
        default_estimators = get_default_estimators(random_state=random_state)
        model_items = [(name, default_estimators[name]) for name in models]
    else:
        model_items = list(models.items())

    return [
        train_classifier(
            estimator,
            x_train,
            y_train,
            x_test,
            y_test,
            model_name=name,
            store_training_data=store_training_data,
            random_state=random_state,
            logger=logger,
        )
        for name, estimator in model_items
    ]


def compare_model_results(
    results: list[ModelTrainingResult],
    *,
    sort_by: str = "auc_pr",
    ascending: bool = False,
) -> pd.DataFrame:
    """Combine model metrics into a comparison table sorted by a chosen metric."""
    if not results:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for result in results:
        row = {"model_name": result.model_name}
        row.update(result.metrics.to_dict())
        rows.append(row)

    comparison = pd.DataFrame(rows)
    if "auc_pr" in comparison.columns:
        comparison["pr_auc"] = comparison["auc_pr"]
    if sort_by in comparison.columns:
        comparison = comparison.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return comparison


def identify_best_model(
    results: list[ModelTrainingResult],
    *,
    metric: str = "auc_pr",
) -> ModelTrainingResult:
    """Return the best model result ranked by the chosen metric."""
    comparison = compare_model_results(results, sort_by=metric)
    best_name = comparison.iloc[0]["model_name"]
    return next(result for result in results if result.model_name == best_name)


def save_model_metrics(
    results: list[ModelTrainingResult] | pd.DataFrame,
    *,
    output_path: Path | None = None,
    output_dir: Path = REPORTS_DIR,
    filename: str = "model_metrics_comparison.csv",
    logger: logging.Logger | None = None,
) -> Path:
    """Persist model comparison metrics to CSV."""
    log = logger or get_logger(__name__)
    report_dir = ensure_dir(output_dir)
    path = output_path or (report_dir / filename)

    if isinstance(results, pd.DataFrame):
        comparison = results
    else:
        comparison = compare_model_results(results)

    comparison.to_csv(path, index=False)
    log.info("Saved model metrics comparison to %s", path)
    return path

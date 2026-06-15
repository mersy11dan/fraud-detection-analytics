"""Hyperparameter tuning for fraud classification models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.config import RANDOM_STATE
from src.utils.logging_config import get_logger

SearchStrategy = Literal["grid", "random"]
SCORING_METRIC = "average_precision"


@dataclass(frozen=True)
class TuningResult:
    """Best estimator and diagnostics from hyperparameter search."""

    model_name: str
    best_estimator: ClassifierMixin
    best_params: dict[str, Any]
    best_cv_score: float
    search_strategy: SearchStrategy


def _logistic_regression_estimator(*, random_state: int) -> LogisticRegression:
    return LogisticRegression(max_iter=1000, random_state=random_state)


def _random_forest_estimator(*, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=random_state, n_jobs=-1)


def _xgboost_estimator(*, random_state: int) -> ClassifierMixin:
    from xgboost import XGBClassifier

    return XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def get_tuning_configs(*, random_state: int = RANDOM_STATE) -> dict[str, dict[str, Any]]:
    """
    Return tuning configuration for each fraud model.

  Each config includes the base estimator, search space, and strategy.
    """
    return {
        "logistic_regression": {
            "estimator": _logistic_regression_estimator(random_state=random_state),
            "search": "grid",
            "param_grid": {
                "C": [0.1, 1.0, 10.0],
                "solver": ["lbfgs"],
            },
        },
        "random_forest": {
            "estimator": _random_forest_estimator(random_state=random_state),
            "search": "random",
            "param_distributions": {
                "n_estimators": [100, 200],
                "max_depth": [12, 20, None],
                "min_samples_leaf": [1, 5],
            },
            "n_iter": 6,
        },
        "xgboost": {
            "estimator": _xgboost_estimator(random_state=random_state),
            "search": "random",
            "param_distributions": {
                "n_estimators": [100, 200],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            "n_iter": 6,
        },
    }


def tune_classifier(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    config: dict[str, Any] | None = None,
    cv: int = 3,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
    logger: logging.Logger | None = None,
) -> TuningResult:
    """
    Tune a classifier with GridSearchCV or RandomizedSearchCV.

    Tuning runs on the provided training split (typically pre-SMOTE) so
    cross-validation reflects natural class prevalence.
    """
    log = logger or get_logger(__name__)
    tuning_config = config or get_tuning_configs(random_state=random_state)[model_name]
    estimator = tuning_config["estimator"]
    strategy: SearchStrategy = tuning_config["search"]

    log.info("Tuning %s with %s search", model_name, strategy)

    if strategy == "grid":
        search = GridSearchCV(
            estimator=estimator,
            param_grid=tuning_config["param_grid"],
            scoring=SCORING_METRIC,
            cv=cv,
            n_jobs=n_jobs,
        )
    elif strategy == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=tuning_config["param_distributions"],
            n_iter=tuning_config.get("n_iter", 10),
            scoring=SCORING_METRIC,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported search strategy: {strategy}")

    search.fit(x_train, y_train)
    log.info(
        "%s best CV AUC-PR: %.4f | params: %s",
        model_name,
        search.best_score_,
        search.best_params_,
    )

    return TuningResult(
        model_name=model_name,
        best_estimator=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
        search_strategy=strategy,
    )

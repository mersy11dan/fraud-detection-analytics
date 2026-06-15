"""Modeling utilities for fraud detection."""

from src.modeling.data import (
    ModelingSplit,
    load_fraud_feature_matrix,
    prepare_fraud_modeling_data,
    stratified_train_test_split,
)
from src.modeling.imbalance import (
    ImbalanceHandlingResult,
    compare_class_distributions,
    prepare_resampled_training_data,
    save_imbalance_report,
)
from src.modeling.metrics import (
    ClassificationMetrics,
    compute_classification_metrics,
    confusion_matrix_frame,
)
from src.modeling.interpretation import (
    describe_feature,
    extract_feature_importance,
    get_coefficient_extremes,
    plot_top_features,
    save_top_features,
)
from src.modeling.reporting import ModelingReportPaths, save_modeling_report
from src.modeling.training import (
    ModelTrainingResult,
    compare_model_results,
    get_default_estimators,
    save_model_metrics,
    train_classifier,
    train_multiple_classifiers,
)

__all__ = [
    "describe_feature",
    "extract_feature_importance",
    "get_coefficient_extremes",
    "ClassificationMetrics",
    "ImbalanceHandlingResult",
    "ModelingReportPaths",
    "ModelTrainingResult",
    "ModelingSplit",
    "compare_class_distributions",
    "compare_model_results",
    "compute_classification_metrics",
    "confusion_matrix_frame",
    "get_default_estimators",
    "load_fraud_feature_matrix",
    "plot_top_features",
    "save_top_features",
    "prepare_fraud_modeling_data",
    "prepare_resampled_training_data",
    "save_imbalance_report",
    "save_model_metrics",
    "save_modeling_report",
    "stratified_train_test_split",
    "train_classifier",
    "train_multiple_classifiers",
]

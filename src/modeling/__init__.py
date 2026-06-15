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
from src.modeling.insights import build_fraud_insights_markdown, save_fraud_insights
from src.modeling.interpretation import (
    describe_feature,
    extract_feature_importance,
    get_coefficient_extremes,
    plot_top_features,
    save_top_features,
)
from src.modeling.reporting import ModelingReportPaths, format_comparison_table, save_modeling_report
from src.modeling.training import (
    ModelTrainingResult,
    compare_model_results,
    get_default_estimators,
    identify_best_model,
    save_model_metrics,
    train_classifier,
    train_multiple_classifiers,
)
from src.modeling.tuning import TuningResult, get_tuning_configs, tune_classifier
from src.modeling.workflow import ModelingWorkflowResult, run_fraud_modeling_workflow

__all__ = [
    "ClassificationMetrics",
    "ImbalanceHandlingResult",
    "ModelingReportPaths",
    "ModelingSplit",
    "ModelingWorkflowResult",
    "ModelTrainingResult",
    "ShapAnalysisPaths",
    "ShapAnalysisResult",
    "TuningResult",
    "build_fraud_insights_markdown",
    "compare_class_distributions",
    "compare_model_results",
    "compute_classification_metrics",
    "confusion_matrix_frame",
    "describe_feature",
    "extract_feature_importance",
    "format_comparison_table",
    "get_coefficient_extremes",
    "get_default_estimators",
    "get_tuning_configs",
    "identify_best_model",
    "load_best_model_for_shap",
    "load_fraud_feature_matrix",
    "plot_top_features",
    "prepare_fraud_modeling_data",
    "prepare_resampled_training_data",
    "run_fraud_modeling_workflow",
    "run_shap_explainability_workflow",
    "save_fraud_insights",
    "save_imbalance_report",
    "save_model_metrics",
    "save_modeling_report",
    "save_top_features",
    "stratified_train_test_split",
    "train_classifier",
    "train_multiple_classifiers",
    "tune_classifier",
]

_LAZY_EXPORTS = {
    "ShapAnalysisPaths": "src.modeling.shap_explain",
    "ShapAnalysisResult": "src.modeling.shap_explain",
    "load_best_model_for_shap": "src.modeling.shap_explain",
    "run_shap_explainability_workflow": "src.modeling.shap_explain",
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

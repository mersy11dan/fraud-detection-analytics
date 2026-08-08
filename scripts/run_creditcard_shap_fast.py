"""Fast creditcard SHAP generation using known best RF hyperparameters.

Avoids full multi-model retuning so SHAP artifacts can be produced quickly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from src.config import CREDITCARD_SHAP_FIGURES_DIR, CREDITCARD_SHAP_REPORTS_DIR, RANDOM_STATE
from src.modeling.data import load_creditcard_feature_matrix, stratified_train_test_split
from src.modeling.shap_explain import run_shap_explainability_workflow
from src.modeling.training import train_classifier
from src.utils.logging_config import get_logger

LOGGER = get_logger(__name__)

# Best params from creditcard RandomizedSearchCV (reports/outputs/creditcard)
BEST_RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 12,
    "min_samples_leaf": 1,
}


def main() -> None:
    features, target = load_creditcard_feature_matrix(logger=LOGGER)
    split = stratified_train_test_split(features, target, random_state=RANDOM_STATE)

    minority = int(split.y_train.sum())
    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=max(1, min(5, minority - 1)),
        sampling_strategy=0.05,
    )
    x_res, y_res = smote.fit_resample(split.x_train, split.y_train)
    x_train = pd.DataFrame(x_res, columns=split.feature_names)
    y_train = pd.Series(y_res, name="class")

    estimator = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **BEST_RF_PARAMS,
    )
    result = train_classifier(
        estimator,
        x_train,
        y_train,
        split.x_test,
        split.y_test,
        model_name="random_forest",
        store_training_data=True,
        random_state=RANDOM_STATE,
        logger=LOGGER,
    )

    analysis = run_shap_explainability_workflow(
        result=result,
        dataset_label="creditcard",
        figures_dir=CREDITCARD_SHAP_FIGURES_DIR,
        reports_dir=CREDITCARD_SHAP_REPORTS_DIR,
        background_size=150,
        explain_size=250,
        random_state=RANDOM_STATE,
        logger=LOGGER,
    )
    print(f"Creditcard SHAP complete: {analysis.model_name}")
    print(f"  figures: {analysis.paths.figures_dir}")
    print(f"  summary: {analysis.paths.summary_markdown}")
    print(analysis.top_features.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

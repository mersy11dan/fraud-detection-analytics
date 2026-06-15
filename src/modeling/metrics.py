"""Evaluation metrics for imbalanced fraud classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Standard metrics for imbalanced binary classification."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    auc_pr: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, float | int]:
        """Return metrics as a plain dictionary."""
        return asdict(self)

    def to_series(self) -> pd.Series:
        """Return metrics as a pandas Series."""
        return pd.Series(self.to_dict())


def compute_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    *,
    pos_label: int = 1,
) -> ClassificationMetrics:
    """
    Compute evaluation metrics appropriate for imbalanced fraud detection.

    ``y_score`` should contain the predicted probability (or score) for the
    positive (fraud) class.
    """
    labels = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    scores = np.asarray(y_score)

    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(precision_score(labels, predictions, pos_label=pos_label, zero_division=0)),
        recall=float(recall_score(labels, predictions, pos_label=pos_label, zero_division=0)),
        f1=float(f1_score(labels, predictions, pos_label=pos_label, zero_division=0)),
        roc_auc=float(roc_auc_score(labels, scores)),
        auc_pr=float(average_precision_score(labels, scores, pos_label=pos_label)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def confusion_matrix_frame(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    """Return a labeled confusion matrix suitable for notebooks and reports."""
    label_order = labels or [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=label_order)
    index = [f"actual_{label}" for label in label_order]
    columns = [f"predicted_{label}" for label in label_order]
    return pd.DataFrame(matrix, index=index, columns=columns)

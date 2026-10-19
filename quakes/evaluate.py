
"""Evaluation metrics for regression and classification models."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_report(y_true, y_pred) -> dict:
    """Return common regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
    }


def classification_report_dict(y_true, y_pred, y_proba) -> dict:
    """Return classification metrics, including ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba, dtype=float)

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)

    # ROC-AUC needs both classes to be present in the true labels.
    if len(np.unique(y_true)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_proba)

    return {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "roc_auc": float(roc_auc),
    }


def best_threshold(y_true, y_proba) -> float:
    """Choose the threshold that maximizes F1 score."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    if len(y_true) != len(y_proba):
        raise ValueError("y_true and y_proba must have the same length")

    if len(y_true) == 0:
        raise ValueError("Inputs cannot be empty")

    # Test probability values as thresholds.
    thresholds = np.unique(y_proba)
    best_score = -1.0
    best_value = 0.5

    for threshold in thresholds:
        predictions = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)

        if score > best_score:
            best_score = score
            best_value = float(threshold)

    return best_value

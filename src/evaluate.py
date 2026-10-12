"""
Evaluation helpers for the fraud detection project.

One function does all the metric computation, so every model from Step 7 to
the final evaluation reports the same numbers in the same way. If metrics are
computed differently in different notebooks, comparisons stop being meaningful.

Usage:

    from src.evaluate import evaluate_model, print_metrics

    metrics = evaluate_model(y_val, y_pred, y_proba)
    print_metrics(metrics, name="Logistic Regression")
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred, y_proba) -> dict:
    """
    Compute the standard metric set for the fraud project.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 = legitimate, 1 = fraud).
    y_pred : array-like of shape (n_samples,)
        Predicted binary labels from a model (already thresholded).
    y_proba : array-like of shape (n_samples,)
        Predicted probability of the positive (fraud) class. Required for
        ROC-AUC and Average Precision.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, roc_auc, average_precision,
        confusion_matrix (as a 2x2 numpy array), n_samples, n_positives.
    """
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy":           accuracy_score(y_true, y_pred),
        "precision":          precision_score(y_true, y_pred, zero_division=0),
        "recall":             recall_score(y_true, y_pred, zero_division=0),
        "f1":                 f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":            roc_auc_score(y_true, y_proba),
        "average_precision":  average_precision_score(y_true, y_proba),
        "confusion_matrix":   cm,
        "n_samples":          len(y_true),
        "n_positives":        int(np.sum(y_true)),
    }


def print_metrics(metrics: dict, name: str = "Model") -> None:
    """Pretty-print a metrics dict in a consistent format."""
    print(f"===== {name} =====")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Precision:           {metrics['precision']:.4f}")
    print(f"  Recall:              {metrics['recall']:.4f}")
    print(f"  F1:                  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:             {metrics['roc_auc']:.4f}")
    print(f"  Average Precision:   {metrics['average_precision']:.4f}")
    print(f"  Samples:             {metrics['n_samples']:,}")
    print(f"  Positives (fraud):   {metrics['n_positives']:,}")
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion matrix:")
    print(f"    TN: {tn:>8,}    FP: {fp:>8,}")
    print(f"    FN: {fn:>8,}    TP: {tp:>8,}")
    print()
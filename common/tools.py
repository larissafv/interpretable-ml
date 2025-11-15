"""Utility functions shared across interpretability methods.

This module currently provides helpers for threshold selection on
precision-recall curves and score normalization used by multiple
explanation algorithms (e.g., PFI, counterfactuals).
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def get_optimal_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute optimal precision/recall/F1 and thresholds per label.

    For each label/column, this function evaluates the precision-recall
    curve and finds the threshold that maximizes F1 = 2PR / (P + R),
    returning the corresponding precision, recall, F1, and threshold.

    Args:
        y_true: Ground-truth binary labels of shape (n_samples, n_labels).
        y_pred: Continuous prediction scores of shape (n_samples, n_labels).

    Returns:
        Tuple of four arrays (each shape: (n_labels,)) containing:
        - precision: Precision at the optimal threshold per label
        - recall: Recall at the optimal threshold per label
        - f1_score: Maximum F1 score per label
        - threshold: Score threshold yielding the maximum F1 per label
    """
    n_labels = y_true.shape[1]
    results = [_optimal_prf1_threshold(y_true[:, k], y_pred[:, k]) for k in range(n_labels)]
    opt_precision, opt_recall, opt_f1_score, opt_threshold = map(
        np.array, zip(*results, strict=False)
    )
    return opt_precision, opt_recall, opt_f1_score, opt_threshold


def _optimal_prf1_threshold(
    y_true_col: np.ndarray, 
    y_pred_col: np.ndarray
) -> tuple[float, float, float, float]:
    """Return precision, recall, f1, and threshold maximizing F1 for one label.

    This is a column-wise helper used by `get_optimal_precision_recall`.
    """
    precision, recall, thresholds = precision_recall_curve(y_true_col, y_pred_col)
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    idx = np.argmax(f1)
    t = thresholds[idx - 1] if idx != 0 else thresholds[0] - 1e-10
    return precision[idx], recall[idx], f1[idx], t


def normalize_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Binarize scores using label-wise optimal thresholds.

    Chooses per-label thresholds that maximize F1 on (y_true, y_pred),
    then returns a binarized prediction matrix.

    Args:
        y_true: Ground-truth binary labels, shape (n_samples, n_labels).
        y_pred: Continuous prediction scores, shape (n_samples, n_labels).

    Returns:
        np.ndarray: Binarized predictions with the same shape as y_pred.
    """
    _, _, _, thresholds = get_optimal_precision_recall(y_true, y_pred)
    return (y_pred > thresholds).astype(int)

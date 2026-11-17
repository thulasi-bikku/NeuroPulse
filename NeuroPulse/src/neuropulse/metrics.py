"""Evaluation metrics and interval estimates used in the manuscript.

Only NumPy and SciPy are required. Functions operate either on label arrays
or on a confusion matrix whose rows are true classes and whose columns are
predicted classes.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy import stats


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= n_classes):
        raise ValueError("labels outside the range [0, n_classes)")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_prf(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall and F1 per class. Undefined ratios are returned as 0."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    col, row = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def summary_scores(cm: np.ndarray) -> dict:
    precision, recall, f1 = per_class_prf(cm)
    n = np.asarray(cm).sum()
    return {
        "accuracy": float(np.trace(cm) / n),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
    }


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Exact two-sided binomial interval."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("require 0 <= successes <= n and n > 0")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, n - successes + 1)
    upper = 1.0 if successes == n else stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lower), float(upper)


def wilson(successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def stratified_bootstrap_accuracy(y_true, y_pred, replicates: int = 2000, confidence: float = 0.95,
                                  seed: int = 0) -> Tuple[float, float]:
    """Percentile interval for accuracy, resampling within each true class."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)]
    out = np.empty(replicates)
    for b in range(replicates):
        idx = np.concatenate([rng.choice(g, size=g.size, replace=True) for g in groups])
        out[b] = correct[idx].mean()
    alpha = 1.0 - confidence
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def expected_calibration_error(probabilities: np.ndarray, y_true: Sequence[int], n_bins: int = 10) -> float:
    """Top-label ECE with equal-width bins on the maximum predicted probability."""
    p = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (conf >= lo) & (conf <= hi) if i == 0 else (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_brier(probabilities: np.ndarray, y_true: Sequence[int]) -> float:
    """Mean over examples of the squared error summed across classes."""
    p = np.asarray(probabilities, dtype=float)
    onehot = np.eye(p.shape[1])[np.asarray(y_true, dtype=int)]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def mcnemar_exact(y_true, pred_a, pred_b) -> Tuple[int, int, float]:
    """Exact McNemar test on paired predictions.

    Returns (b, c, p). b counts examples only model A classified correctly,
    c counts examples only model B classified correctly, and p is the
    two-sided exact binomial p-value under equal error rates.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def confidence_floor_for_errors(mean_confidence: float, n: int, n_errors: int) -> float:
    """Lower bound on the mean top score of the wrong predictions.

    Assumes all n - n_errors correct predictions had score 1, the most
    favourable case for the correct predictions.
    """
    if n_errors <= 0:
        raise ValueError("n_errors must be positive")
    return float((mean_confidence * n - (n - n_errors)) / n_errors)

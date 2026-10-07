"""Metric functions on plain arrays: one group's labels and scores in, one number out.

Inputs are array-likes (lists, numpy arrays, pandas Series) of equal length:
y_true holds 0/1 labels, y_score holds predicted probabilities. These
functions do not re-validate the values: audit() does that once, up front.

They must handle empty and one-class inputs, because single groups and
bootstrap replicates produce them. A metric whose denominator is 0 is
undefined and comes back as NaN, never as an error or a made-up number.
All results are plain Python floats and ints, not numpy scalars.
"""

import math

import numpy as np
from numpy.typing import ArrayLike


def _as_pair(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    if y.ndim != 1 or s.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D")
    if len(y) != len(s):
        raise ValueError(f"y_true and y_score must have the same length, got {len(y)} and {len(s)}")
    return y, s


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else math.nan


def _check_threshold(threshold: float) -> None:
    is_number = isinstance(threshold, (int, float, np.integer, np.floating)) and not isinstance(
        threshold, bool
    )
    if not (is_number and 0 <= threshold <= 1):
        raise ValueError(f"threshold must be a number in [0, 1], got {threshold!r}")


# ---------------------------------------------------------------- threshold-free


def auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Chance that a random positive scores above a random negative (ties count half).

    NaN when the group has no positives or no negatives.
    """
    y, s = _as_pair(y_true, y_score)
    pos = y == 1
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    # Average ranks, so a tied (positive, negative) pair counts as half a win.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)  # rank of the last copy of each distinct score
    avg_rank = upper - (counts - 1) / 2  # tied copies share their middle rank
    ranks = avg_rank[inverse]
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Mean squared error of the predicted probability. Lower is better. NaN when empty."""
    y, s = _as_pair(y_true, y_score)
    if len(y) == 0:
        return math.nan
    return float(np.mean((s - y) ** 2))


def o_e(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Observed over expected positives. Above 1 means the model underestimates the group.

    NaN when the expected count (sum of scores) is 0.
    """
    y, s = _as_pair(y_true, y_score)
    return _safe_div(y.sum(), s.sum())


# ---------------------------------------------------------------- threshold metrics


def confusion_counts(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn), where a patient is flagged when y_score >= threshold."""
    _check_threshold(threshold)
    y, s = _as_pair(y_true, y_score)
    flagged = s >= threshold
    positive = y == 1
    tp = int(np.sum(flagged & positive))
    fp = int(np.sum(flagged & ~positive))
    fn = int(np.sum(~flagged & positive))
    tn = int(np.sum(~flagged & ~positive))
    return tp, fp, fn, tn


def threshold_rates(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, float]:
    """Return flag_rate, fpr, fnr and ppv at threshold. Each is NaN when its denominator is 0."""
    tp, fp, fn, tn = confusion_counts(y_true, y_score, threshold)
    return {
        "flag_rate": _safe_div(tp + fp, tp + fp + fn + tn),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "ppv": _safe_div(tp, tp + fp),
    }

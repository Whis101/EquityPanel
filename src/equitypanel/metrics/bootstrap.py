"""Stratified bootstrap: resample patients within each group, then take percentile CIs.

Resampling within each group keeps every group at its real size, so small groups
get honestly wide intervals. One seeded numpy Generator drives all draws, so the
same inputs and seed always give the same intervals.
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

Stat = Callable[[np.ndarray, np.ndarray], dict[str, float]]


def _check_ci(ci: float) -> None:
    is_number = isinstance(ci, (int, float, np.integer, np.floating)) and not isinstance(ci, bool)
    if not (is_number and 0 < ci < 1):
        raise ValueError(f"ci must be a number strictly between 0 and 1, got {ci!r}")


def _check_n_boot(n_boot: int) -> None:
    is_int = isinstance(n_boot, (int, np.integer)) and not isinstance(n_boot, bool)
    if not (is_int and n_boot >= 1):
        raise ValueError(f"n_boot must be an int >= 1, got {n_boot!r}")


def percentile_ci(replicates: ArrayLike, ci: float = 0.95) -> tuple[float, float]:
    """Middle `ci` share of the defined (non-NaN) replicates, as (low, high).

    (NaN, NaN) when fewer than half the replicates are defined: the rest would be
    a biased subset. Raises ValueError unless 0 < ci < 1.
    """
    _check_ci(ci)
    values = np.asarray(replicates, dtype=float)
    defined = values[~np.isnan(values)]
    if len(defined) == 0 or len(defined) < len(values) / 2:
        return math.nan, math.nan
    low, high = np.percentile(defined, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(low), float(high)


def stratified_replicates(
    y_true: ArrayLike,
    y_score: ArrayLike,
    groups: ArrayLike,
    stat: Stat,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Run `stat` on n_boot resamples of each group. Returns {group: {name: values}}.

    Draw order (pinned by tests): one Generator from `seed`; for each replicate,
    groups in sorted order, each drawing rng.integers(0, n_g, n_g). Undefined
    results stay as NaN, so every array has length n_boot.
    """
    _check_n_boot(n_boot)
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    if not (len(y) == len(s) == len(g)):
        raise ValueError("y_true, y_score and groups must have the same length")

    rng = np.random.default_rng(seed)
    keys = np.unique(g).tolist()
    data = {k: (y[g == k], s[g == k]) for k in keys}
    out = {k: {} for k in keys}
    for b in range(n_boot):
        for k in keys:
            y_k, s_k = data[k]
            idx = rng.integers(0, len(y_k), len(y_k))
            for name, value in stat(y_k[idx], s_k[idx]).items():
                out[k].setdefault(name, np.full(n_boot, np.nan))[b] = value
    return out

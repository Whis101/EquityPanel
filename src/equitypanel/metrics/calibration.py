"""Calibration curves with equal-count (quantile) bins.

Risk scores bunch up near 0, so equal-width bins would leave most bins nearly
empty. Quantile bins give each point on the curve about the same number of
patients. Tied scores always share one bin, so bins can come out uneven, and
bins that end up empty are dropped.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from equitypanel.metrics.core import _as_pair

CURVE_COLUMNS = ("bin", "mean_pred", "obs_rate", "n")


def _empty_curve() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bin": pd.Series(dtype="int64"),
            "mean_pred": pd.Series(dtype="float64"),
            "obs_rate": pd.Series(dtype="float64"),
            "n": pd.Series(dtype="int64"),
        }
    )


def _check_n_bins(n_bins: int) -> None:
    is_int = isinstance(n_bins, (int, np.integer)) and not isinstance(n_bins, bool)
    if not (is_int and n_bins >= 1):
        raise ValueError(f"n_bins must be an int >= 1, got {n_bins!r}")


def calibration_curve(y_true: ArrayLike, y_score: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Bin patients by predicted risk and compare mean prediction to observed rate per bin.

    Returns a DataFrame with columns bin (numbered from 0, no gaps), mean_pred,
    obs_rate and n, one row per non-empty bin in increasing order of risk.
    Raises ValueError if n_bins is not an int >= 1.
    """
    _check_n_bins(n_bins)
    y, s = _as_pair(y_true, y_score)
    if len(y) == 0:
        return _empty_curve()

    # Duplicate edges (from ties) collapse, merging bins instead of splitting a tie.
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    # side="right": a score sitting on an inner edge goes to the upper bin.
    bin_id = np.searchsorted(edges[1:-1], s, side="right")

    curve = (
        pd.DataFrame({"bin_id": bin_id, "y": y, "s": s})
        .groupby("bin_id", sort=True)
        .agg(mean_pred=("s", "mean"), obs_rate=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    curve.insert(0, "bin", np.arange(len(curve), dtype="int64"))
    curve["n"] = curve["n"].astype("int64")
    return curve[list(CURVE_COLUMNS)]

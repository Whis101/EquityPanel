"""Per-group flag rate, FPR, FNR and PPV across many flagging cutoffs (numpy + pandas only).

A program can only follow up so many patients. capacity_thresholds() turns "flag the top
x% of patients" into score cutoffs; threshold_sweep() then shows, for every group, who is
flagged and who is missed at each cutoff, so the effect of the cutoff choice is visible.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from equitypanel.data.schema import validate_audit_frame
from equitypanel.metrics.core import threshold_rates

SWEEP_COLUMNS = ("group", "flag_share", "threshold", "n", "flag_rate", "fpr", "fnr", "ppv")


def capacity_thresholds(scores: Sequence[float], shares: Sequence[float]) -> np.ndarray:
    """Score cutoffs that flag (about) each share of patients: the (1 - share) quantile.

    A patient is flagged when score >= cutoff, matching metrics.core. Raises ValueError
    for a share outside (0, 1].
    """
    shares = np.asarray(shares, dtype=float)
    if ((shares <= 0) | (shares > 1)).any():
        raise ValueError("every share must be in (0, 1]")
    return np.quantile(np.asarray(scores, dtype=float), 1 - shares)


def threshold_sweep(
    df: pd.DataFrame, group_col: str, shares: Sequence[float] = tuple(np.arange(1, 51) / 100)
) -> pd.DataFrame:
    """One row per (group, share): the cutoff and the group's flag rate, FPR, FNR and PPV.

    df is an audit frame (y_true 0/1, y_score in [0, 1], group_col). Cutoffs come from
    all patients' scores, so every group faces the same cutoff, as in a real program.
    Rates that are undefined for a group (e.g. FNR with no readmissions) are NaN.
    """
    validate_audit_frame(df, [group_col])
    cutoffs = capacity_thresholds(df["y_score"], shares)
    rows = []
    for group, part in df.groupby(group_col, sort=True):
        y, s = part["y_true"].to_numpy(), part["y_score"].to_numpy()
        for share, cutoff in zip(shares, cutoffs, strict=True):
            rates = threshold_rates(y, s, float(cutoff))
            rows.append(
                {
                    "group": str(group),
                    "flag_share": float(share),
                    "threshold": float(cutoff),
                    "n": len(part),
                    **{k: rates[k] for k in ("flag_rate", "fpr", "fnr", "ppv")},
                }
            )
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))

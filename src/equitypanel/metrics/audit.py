"""audit(): the per-group report card for an audit frame.

For each group column (one at a time) and each group in it, compute every metric
on the real data, add stratified bootstrap CIs, and add gaps against a reference
group. Output is two tidy (long) tables: metrics and calibration curves.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from equitypanel.data.schema import validate_audit_frame
from equitypanel.metrics.bootstrap import (
    _check_ci,
    _check_n_boot,
    percentile_ci,
    stratified_replicates,
)
from equitypanel.metrics.calibration import CURVE_COLUMNS, _check_n_bins, calibration_curve
from equitypanel.metrics.core import _check_threshold, _safe_div, auc, brier, o_e, threshold_rates

METRIC_COLUMNS = (
    "group_col",
    "group",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "small_group",
    "n_undefined",
)
CALIBRATION_COLUMNS = ("group_col", "group", *CURVE_COLUMNS)

# Bootstrapped metrics, in output order. Counts (n, n_pos) come first and get no CI.
CI_METRICS = ("prevalence", "auc", "brier", "o_e", "flag_rate", "fpr", "fnr", "ppv")
# A difference in base rates isn't a model error, so prevalence gets no gap.
GAP_METRICS = CI_METRICS[1:]


@dataclass(frozen=True)
class AuditReport:
    """metrics: one row per (group_col, group, metric). calibration: one row per bin."""

    metrics: pd.DataFrame
    calibration: pd.DataFrame


def _group_stats(y: np.ndarray, s: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "prevalence": _safe_div(y.sum(), len(y)),
        "auc": auc(y, s),
        "brier": brier(y, s),
        "o_e": o_e(y, s),
        **threshold_rates(y, s, threshold),
    }


def _row(
    base: dict, metric: str, value: float, replicates: np.ndarray | None = None, ci: float = 0.95
) -> dict:
    """One metrics row. Counts pass no replicates and get no CI."""
    if replicates is None:
        low, high, n_undefined = math.nan, math.nan, 0
    else:
        low, high = percentile_ci(replicates, ci)
        n_undefined = int(np.isnan(replicates).sum())
    return {
        **base,
        "metric": metric,
        "value": float(value),
        "ci_low": low,
        "ci_high": high,
        "n_undefined": n_undefined,
    }


def _check_min_group_size(min_group_size: int) -> None:
    is_int = isinstance(min_group_size, (int, np.integer)) and not isinstance(min_group_size, bool)
    if not (is_int and min_group_size >= 1):
        raise ValueError(f"min_group_size must be an int >= 1, got {min_group_size!r}")


def _resolve_references(
    reference: Mapping[str, object] | None, group_cols: Sequence[str]
) -> dict[str, str]:
    reference = dict(reference or {})
    unknown = [col for col in reference if col not in group_cols]
    if unknown:
        raise ValueError(f"reference names columns not in group_cols: {unknown}")
    return {col: str(value) for col, value in reference.items()}


def audit(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    threshold: float,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = 10,
    reference: Mapping[str, object] | None = None,
    min_group_size: int = 30,
    ci: float = 0.95,
) -> AuditReport:
    """Audit a risk model's scores for gaps across groups, one group column at a time.

    df is an audit frame (y_true 0/1, y_score in [0, 1], plus group_cols). A patient
    is flagged when y_score >= threshold. The reference group for gaps is the largest
    group (ties: first in sorted order) unless `reference` maps a column to a group.
    Groups below min_group_size are flagged, not dropped. Undefined metrics are NaN.

    Raises SchemaError for an invalid frame and ValueError for a bad option.
    """
    validate_audit_frame(df, group_cols)
    _check_threshold(threshold)
    _check_n_boot(n_boot)
    _check_ci(ci)
    _check_n_bins(n_bins)
    _check_min_group_size(min_group_size)
    references = _resolve_references(reference, group_cols)

    y = df["y_true"].to_numpy(dtype=float)
    s = df["y_score"].to_numpy(dtype=float)
    stat = partial(_group_stats, threshold=threshold)
    metric_rows: list[dict] = []
    curves: list[pd.DataFrame] = []

    for col in group_cols:
        groups = df[col].astype(str).to_numpy()
        keys = sorted(set(groups))
        sizes = {k: int(np.sum(groups == k)) for k in keys}
        ref = references.get(col, max(keys, key=sizes.__getitem__))
        if ref not in sizes:
            raise ValueError(f"reference group {ref!r} not found in column {col!r}")

        # A fresh generator per column: adding a column never changes another's CIs.
        reps = stratified_replicates(y, s, groups, stat, n_boot=n_boot, seed=seed)
        point = {k: stat(y[groups == k], s[groups == k]) for k in keys}

        for k in keys:
            base = {"group_col": col, "group": k, "n": sizes[k]}
            base["small_group"] = sizes[k] < min_group_size
            metric_rows.append(_row(base, "n", sizes[k]))
            metric_rows.append(_row(base, "n_pos", y[groups == k].sum()))
            for m in CI_METRICS:
                metric_rows.append(_row(base, m, point[k][m], reps[k][m], ci))
            if k != ref:  # the reference group has no gap rows
                for m in GAP_METRICS:
                    gap = point[k][m] - point[ref][m]
                    gap_reps = reps[k][m] - reps[ref][m]  # NaN if either side is NaN
                    metric_rows.append(_row(base, f"{m}_gap", gap, gap_reps, ci))

            curve = calibration_curve(y[groups == k], s[groups == k], n_bins=n_bins)
            curve.insert(0, "group", k)
            curve.insert(0, "group_col", col)
            curves.append(curve)

    metrics = pd.DataFrame(metric_rows, columns=list(METRIC_COLUMNS)).astype(
        {
            "group_col": "object",
            "group": "object",
            "metric": "object",
            "value": "float64",
            "ci_low": "float64",
            "ci_high": "float64",
            "n": "int64",
            "small_group": "bool",
            "n_undefined": "int64",
        }
    )
    calibration = pd.concat(curves, ignore_index=True)[list(CALIBRATION_COLUMNS)]
    return AuditReport(metrics=metrics, calibration=calibration)

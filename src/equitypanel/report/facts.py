"""The facts JSON: every number the report states, each under a stable ID.

This is the only input the LLM summary ever sees. It holds aggregate numbers (group
sizes, metrics, confidence intervals), never patient rows. IDs look like
"race.Black.fnr", "model.auc_test" or "egfr.Black.waitlist_le20_2021".
"""

import math

import pandas as pd

from equitypanel.reclassification.reclassify import RACE_TERM_COL

DECIMALS = 4  # facts are rounded so the LLM and the checker see the same numbers

METRIC_LABELS = {
    "n": "patients",
    "n_pos": "readmitted patients",
    "prevalence": "readmission rate",
    "auc": "AUC (ranking accuracy, 0.5 = chance)",
    "brier": "Brier score (lower is better)",
    "o_e": "observed / expected readmissions (1 = calibrated)",
    "flag_rate": "share flagged for follow-up",
    "fpr": "false positive rate (flagged but not readmitted)",
    "fnr": "false negative rate (readmitted but not flagged)",
    "ppv": "positive predictive value (flagged who were readmitted)",
}
MODEL_LABELS = {
    "n_test": "patients in the audited test set",
    "prevalence_test": "readmission rate in the test set",
    "auc_test": "overall AUC on the test set",
    "threshold": "risk score at or above which a patient is flagged",
    "flag_share_test": "share of test patients flagged",
}
EGFR_KEYS = (
    "n",
    "n_lower",
    "n_higher",
    "ckd_lt60_2009",
    "ckd_lt60_2021",
    "referral_lt30_2009",
    "referral_lt30_2021",
    "waitlist_le20_2009",
    "waitlist_le20_2021",
    "waitlist_le20_newly_below",
    "waitlist_le20_newly_above",
)


def _num(value) -> float | int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if float(value).is_integer() and abs(value) >= 1:
        return int(value)
    return round(float(value), DECIMALS)


def reference_groups(metrics: pd.DataFrame) -> dict[str, str]:
    """The reference group of each group column: the one group with no *_gap rows."""
    refs = {}
    gaps = metrics[metrics["metric"].str.endswith("_gap")]
    for col, part in metrics.groupby("group_col", sort=False):
        with_gap = set(gaps.loc[gaps["group_col"] == col, "group"])
        without = [g for g in part["group"].unique() if g not in with_gap]
        if len(without) != 1:
            raise ValueError(f"cannot tell the reference group of {col!r}: {without}")
        refs[col] = str(without[0])
    return refs


def _metric_fact(row: pd.Series, ref: str) -> dict:
    metric = row["metric"]
    base = metric.removesuffix("_gap")
    label = METRIC_LABELS.get(base, base)
    if metric.endswith("_gap"):
        label = f"gap in {label} vs the reference group ({ref})"
    fact = {
        "label": label,
        "value": _num(row["value"]),
        "n": int(row["n"]),
        "small_group": bool(row["small_group"]),
    }
    low, high = _num(row["ci_low"]), _num(row["ci_high"])
    if low is not None and high is not None:
        fact["ci_95"] = [low, high]
    return fact


def build_facts(
    metrics: pd.DataFrame,
    model_info: dict | None = None,
    egfr_summary: pd.DataFrame | None = None,
    *,
    dataset: str = "UCI Diabetes 130-US Hospitals (public)",
) -> dict:
    """Collect every reportable number into {"context": ..., "facts": {id: fact}}.

    metrics is the audit() metrics frame; model_info the baseline summary
    (uci_model.json); egfr_summary a reclassification_summary() frame. Missing
    (NaN) values are kept as null so the summary can't invent them.
    """
    refs = reference_groups(metrics)
    facts: dict[str, dict] = {}
    for _, row in metrics.iterrows():
        col, group = row["group_col"], str(row["group"])
        facts[f"{col}.{group}.{row['metric']}"] = _metric_fact(row, refs[col])

    if model_info:
        for key, label in MODEL_LABELS.items():
            if key in model_info:
                facts[f"model.{key}"] = {"label": label, "value": _num(model_info[key])}

    context = {
        "dataset": dataset,
        "task": "predict 30-day hospital readmission; the top-risk patients are flagged",
        "reference_groups": refs,
        "small_group_rule": "fewer than 30 readmitted or fewer than 30 not-readmitted patients",
        "ci": "95% stratified bootstrap",
    }
    if egfr_summary is not None:
        rows = egfr_summary[egfr_summary["group_col"].isin(["all", RACE_TERM_COL])]
        for _, row in rows.iterrows():
            for key in EGFR_KEYS:
                facts[f"egfr.{row['group']}.{key}"] = {"value": _num(row[key])}
        context["egfr"] = (
            "SYNTHETIC Synthea patients, not real-world rates. CKD-EPI 2009 (race-based) vs "
            "2021 (race-free). Lines: ckd_lt60 = eGFR < 60 (CKD diagnosis), referral_lt30 = "
            "eGFR < 30 (specialist referral), waitlist_le20 = eGFR <= 20 (transplant waiting "
            "time can start). newly_below = below the line under 2021 only; n_lower = moved "
            "to a worse stage under 2021."
        )
    return {"context": context, "facts": facts}

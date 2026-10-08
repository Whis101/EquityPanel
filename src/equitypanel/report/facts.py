"""The facts JSON: every number the report states, each under a stable ID.

This is the only input the LLM summary ever sees. It holds aggregate numbers (group
sizes, metrics, confidence intervals), never patient rows. IDs look like
"race.Black.fnr", "model.auc_test" or "egfr.Black.waitlist_le20_2021".

Aggregates are not automatically de-identified: a group of 2 patients reveals their
outcomes. suppress_small_cells() removes small groups before anything is sent out.
"""

import math

import pandas as pd

from equitypanel.reclassification.reclassify import RACE_TERM_COL

DECIMALS = 4  # facts are rounded so the LLM and the checker see the same numbers
MIN_CELL = 11  # CMS small-cell convention: never release counts of 1-10

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
) -> dict:
    """Collect every reportable number into {"context": ..., "facts": {id: fact}}.

    metrics is the audit() metrics frame; model_info the baseline summary
    (uci_model.json); egfr_summary a reclassification_summary() frame. Missing
    (NaN) values are kept as null so the summary can't invent them. No dataset or
    institution name is included, so it can't leak into the LLM payload.
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
            "eGFR < 30 (a line commonly used for specialist referral), waitlist_le20 = "
            "eGFR <= 20 (transplant waiting time can start). newly_below = below the line "
            "under 2021 only; n_lower = moved to a worse stage under 2021."
        )
    return {"context": context, "facts": facts}


def _is_small(count, min_cell: int) -> bool:
    return count is not None and 1 <= count < min_cell


def suppress_small_cells(facts_json: dict, min_cell: int = MIN_CELL) -> tuple[dict, list[str]]:
    """Remove groups whose patients, events or non-events number 1 to min_cell - 1.

    Returns (new facts JSON, suppressed groups as "group_col.group"). Following the
    CMS cell-size convention:
    - a small group loses all its facts (counts, rates, AUC, CIs, gaps);
    - if it is the only small group in its column, the next-smallest group is
      suppressed too, so it can't be recovered by subtracting from the others;
    - when anything is suppressed, overall totals (test-set size and readmission
      rate) are dropped for the same reason;
    - eGFR counts of 1 to min_cell - 1 are removed one by one.
    This lowers re-identification risk; it does not remove it. The input is not modified.
    """
    facts = dict(facts_json["facts"])
    groups: dict[str, dict[str, int]] = {}
    for fid, fact in facts.items():
        col, _, rest = fid.partition(".")
        group, _, metric = rest.rpartition(".")
        if col in ("model", "egfr") or metric not in ("n", "n_pos"):
            continue
        groups.setdefault(col, {}).setdefault(group, {})[metric] = fact["value"]

    suppressed: list[str] = []
    for col, by_group in groups.items():
        small = []
        for group, counts in by_group.items():
            n, n_pos = counts.get("n"), counts.get("n_pos")
            n_neg = n - n_pos if n is not None and n_pos is not None else None
            if any(_is_small(c, min_cell) for c in (n, n_pos, n_neg)):
                small.append(group)
        if len(small) == 1 and len(by_group) > 1:
            others = sorted((g for g in by_group if g not in small), key=lambda g: by_group[g]["n"])
            small.append(others[0])
        suppressed += [f"{col}.{g}" for g in small]

    if suppressed:
        prefixes = tuple(f"{s}." for s in suppressed)
        facts = {fid: f for fid, f in facts.items() if not fid.startswith(prefixes)}
        for key in ("model.n_test", "model.prevalence_test"):
            facts.pop(key, None)
    for fid in [f for f in facts if f.startswith("egfr.")]:
        value = facts[fid]["value"]
        if isinstance(value, int) and _is_small(value, min_cell):
            del facts[fid]
            suppressed.append(fid)

    context = dict(facts_json["context"])
    context["small_cell_suppression"] = {
        "min_cell": min_cell,
        "suppressed": suppressed,
        "rule": f"groups or counts of 1 to {min_cell - 1} patients or events were removed",
    }
    return {"context": context, "facts": facts}, suppressed

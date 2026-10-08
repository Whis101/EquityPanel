"""Build the self-contained HTML report. Needs the [report] extra (jinja2, matplotlib).

Every value inserted into the page is escaped (Jinja2 autoescape), so a group named
"<script>" is shown as text. Only the SVG charts, which matplotlib escapes itself,
are inserted as markup.
"""

from datetime import date

import pandas as pd

from equitypanel import __version__
from equitypanel.data.schema import AGE_BANDS
from equitypanel.report.charts import calibration_chart, figure_to_svg, group_ci_chart
from equitypanel.report.facts import reference_groups
from equitypanel.report.summary import Summary

TABLE_METRICS = ("auc", "brier", "o_e", "fpr", "fnr", "ppv")
PERCENT_METRICS = {"prevalence", "flag_rate", "fpr", "fnr", "ppv"}
METRIC_HEADERS = {
    "auc": "AUC",
    "brier": "Brier",
    "o_e": "O/E",
    "fpr": "FPR",
    "fnr": "FNR",
    "ppv": "PPV",
    "prevalence": "Readmitted",
}
GROUP_COL_TITLES = {"race": "Race", "sex": "Sex", "age_band": "Age band"}
CHART_METRICS = ("auc", "fnr")


def fmt(metric: str, value) -> str:
    """Format a metric value for display: percentages for rates, 3 decimals otherwise."""
    if value is None or pd.isna(value):
        return "n/a"
    base = metric.removesuffix("_gap")
    if base in PERCENT_METRICS:
        sign = "+" if metric.endswith("_gap") and value > 0 else ""
        return f"{sign}{100 * value:.1f}%"
    if metric.endswith("_gap"):
        return f"{value:+.3f}"
    if base in ("n", "n_pos"):
        return f"{int(value):,}"
    return f"{value:.3f}"


def order_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows so age bands follow AGE_BANDS ("<50" first); other groups keep their order."""
    rank = {band: i for i, band in enumerate(AGE_BANDS)}
    key = [
        rank.get(str(g), len(rank)) if col == "age_band" else 0
        for col, g in zip(df["group_col"], df["group"], strict=True)
    ]
    return df.assign(_order=key).sort_values("_order", kind="stable").drop(columns="_order")


def _cell(row: pd.Series | None, metric: str) -> dict:
    if row is None:
        return {"value": "n/a", "ci": ""}
    ci = ""
    if pd.notna(row["ci_low"]) and pd.notna(row["ci_high"]):
        ci = f"{fmt(metric, row['ci_low'])} – {fmt(metric, row['ci_high'])}"
    return {"value": fmt(metric, row["value"]), "ci": ci}


def _group_section(metrics: pd.DataFrame, calibration: pd.DataFrame, col: str, ref: str) -> dict:
    part = metrics[metrics["group_col"] == col]
    lookup = {(str(r["group"]), r["metric"]): r for _, r in part.iterrows()}
    rows = []
    for group in dict.fromkeys(part["group"].astype(str)):
        n_row = lookup[(group, "n")]
        rows.append(
            {
                "group": group,
                "is_reference": group == ref,
                "small": bool(n_row["small_group"]),
                "n": fmt("n", n_row["value"]),
                "n_pos": fmt("n_pos", lookup[(group, "n_pos")]["value"])
                if (group, "n_pos") in lookup
                else "n/a",
                "cells": [_cell(lookup.get((group, m)), m) for m in TABLE_METRICS],
                "gaps": [
                    fmt(f"{m}_gap", lookup[(group, f"{m}_gap")]["value"])
                    if (group, f"{m}_gap") in lookup
                    else "—"
                    for m in TABLE_METRICS
                ],
            }
        )
    charts = [group_ci_chart(metrics, col, m) for m in CHART_METRICS]
    if not calibration[calibration["group_col"] == col].empty:
        charts.append(calibration_chart(calibration, col))
    return {
        "col": col,
        "title": GROUP_COL_TITLES.get(col, col),
        "reference": ref,
        "rows": rows,
        "charts": charts,
    }


def _egfr_section(egfr_summary: pd.DataFrame) -> dict:
    from equitypanel.reclassification.plot import SYNTHETIC_LABEL, headline_chart
    from equitypanel.reclassification.reclassify import LINES, RACE_TERM_COL

    keep = egfr_summary[egfr_summary["group_col"].isin(["all", RACE_TERM_COL, "race"])]
    rows = []
    for _, r in keep.iterrows():
        rows.append(
            {
                "group": "All patients" if r["group_col"] == "all" else str(r["group"]),
                "kind": r["group_col"],
                "small": bool(r["small_group"]),
                "n": f"{int(r['n']):,}",
                "lower": f"{int(r['n_lower']):,}",
                "higher": f"{int(r['n_higher']):,}",
                "lines": [
                    f"{int(r[f'{line}_2009']):,} → {int(r[f'{line}_2021']):,}" for line in LINES
                ],
            }
        )
    return {
        "label": SYNTHETIC_LABEL,
        "chart": figure_to_svg(headline_chart(egfr_summary)),
        "rows": rows,
    }


def _summary_context(summary: Summary | None) -> dict | None:
    if summary is None:
        return None
    return {
        "findings": summary.findings,
        "rejected": summary.rejected,
        "model": summary.model,
        "attempts": summary.attempts,
        "error": summary.error,
    }


def build_report(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    model_info: dict | None = None,
    egfr_summary: pd.DataFrame | None = None,
    summary: Summary | None = None,
    *,
    title: str = "EquityPanel audit report",
    dataset: str = "UCI Diabetes 130-US Hospitals (public data)",
    methods: dict | None = None,
    generated: date | None = None,
) -> str:
    """Return the full report as one self-contained HTML string.

    metrics / calibration come from audit(); model_info is the baseline summary dict;
    egfr_summary a reclassification_summary() frame (shown labelled as synthetic);
    summary a checked LLM Summary, or None for "not generated".
    """
    from jinja2 import Environment, PackageLoader, select_autoescape

    metrics = order_groups(metrics)
    calibration = order_groups(calibration)
    refs = reference_groups(metrics)
    env = Environment(
        loader=PackageLoader("equitypanel.report", "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    small_groups = (
        metrics[(metrics["metric"] == "n") & metrics["small_group"]]
        .apply(
            lambda r: f"{GROUP_COL_TITLES.get(r['group_col'], r['group_col'])}: {r['group']}",
            axis=1,
        )
        .tolist()
    )
    key_numbers = []
    if model_info:
        key_numbers = [
            ("Patients audited", f"{int(model_info['n_test']):,}"),
            ("Overall AUC", f"{model_info['auc_test']:.3f}"),
            ("Readmitted within 30 days", f"{100 * model_info['prevalence_test']:.1f}%"),
            ("Flagged for follow-up", f"{100 * model_info['flag_share_test']:.1f}%"),
        ]
    key_numbers.append(("Small groups (uncertain)", str(len(small_groups))))

    template = env.get_template("report.html.j2")
    return template.render(
        title=title,
        dataset=dataset,
        version=__version__,
        generated=(generated or date.today()).isoformat(),
        key_numbers=key_numbers,
        small_groups=small_groups,
        summary=_summary_context(summary),
        headers=[METRIC_HEADERS[m] for m in TABLE_METRICS],
        sections=[_group_section(metrics, calibration, col, refs[col]) for col in refs],
        egfr=_egfr_section(egfr_summary) if egfr_summary is not None else None,
        model_info=model_info,
        methods=methods or {},
    )

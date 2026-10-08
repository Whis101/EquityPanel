"""EquityPanel interactive demo (Streamlit), on bundled public and synthetic data only.

Run locally (Git Bash, repo root):  streamlit run demo/app.py
No uploads, no API calls: the AI summary shown is the saved, checked one, and the
"try the number check" box runs EquityPanel's guardrail locally.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # works without `pip install -e .` too
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from equitypanel.metrics.audit import audit  # noqa: E402
from equitypanel.metrics.core import auc, threshold_rates  # noqa: E402
from equitypanel.report.summary import check_finding  # noqa: E402
from equitypanel.thresholds import capacity_thresholds, threshold_sweep  # noqa: E402

DATA = ROOT / "demo" / "data"
REPO = "https://github.com/Whis101/EquityPanel"
REPORT = "https://whis101.github.io/EquityPanel/"
GROUP_COLS = {"Race": "race", "Sex": "sex", "Age band": "age_band"}
AGE_ORDER = ["<50", "50-59", "60-69", "70-79", "80+"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METRICS = {
    "False negative rate (readmitted but not flagged)": "fnr",
    "Flag rate (share flagged for follow-up)": "flag_rate",
    "False positive rate (flagged but not readmitted)": "fpr",
    "Positive predictive value (flagged who were readmitted)": "ppv",
}
SMALL_EVENTS = 30  # EquityPanel's small-group rule
MIN_CELL = 11

INTENDED_USE = (
    "**Research and education demo, not a clinical tool.** Numbers describe groups in a public "
    "dataset (UCI, 1999-2008 US hospitals) and synthetic patients, not individuals. Not medical "
    "advice, not clinically validated, and not for decisions about individual patients. "
    "EquityPanel reports performance gaps; it does not certify a model as fair, safe or compliant."
)
CREDITS = (
    "Readmission data: Clore J, Cios K, DeShazo J, Strack B (2014). Diabetes 130-US Hospitals for "
    "Years 1999-2008 [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5230J, licensed under CC BY 4.0; scores here are from EquityPanel's "
    "baseline model. Synthetic patients: Synthea (The MITRE Corporation); Walonoski J, et al. "
    "JAMIA 2018. EquityPanel is independent and not affiliated with any organization named here. "
    "MIT License, provided as is, without warranty."
)


@st.cache_data
def load_scores() -> pd.DataFrame:
    return pd.read_csv(DATA / "uci_test_scores.csv", dtype={"age_band": str})


@st.cache_data
def load_json(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def ordered(groups) -> list[str]:
    groups = [str(g) for g in groups]
    if set(groups) <= set(AGE_ORDER):
        return [g for g in AGE_ORDER if g in groups]
    return sorted(groups)


@st.cache_data
def sweep(group_col: str) -> pd.DataFrame:
    return threshold_sweep(load_scores(), group_col)


@st.cache_data
def point_table(group_col: str, share: float) -> tuple[pd.DataFrame, float]:
    df = load_scores()
    cutoff = float(capacity_thresholds(df["y_score"], [share])[0])
    rows = []
    for group in ordered(df[group_col].unique()):
        part = df[df[group_col] == group]
        y, s = part["y_true"], part["y_score"]
        n_pos = int(y.sum())
        rates = threshold_rates(y, s, cutoff)
        note = []
        if min(len(part), n_pos, len(part) - n_pos) < MIN_CELL:
            note.append("very small")
        elif min(n_pos, len(part) - n_pos) < SMALL_EVENTS:
            note.append("small group")
        rows.append(
            {
                "Group": group,
                "Patients": len(part),
                "Readmitted": n_pos,
                "AUC": auc(y, s),
                "Flagged": rates["flag_rate"],
                "Missed (FNR)": rates["fnr"],
                "False alarms (FPR)": rates["fpr"],
                "PPV": rates["ppv"],
                "Note": ", ".join(note),
            }
        )
    return pd.DataFrame(rows), cutoff


@st.cache_data(show_spinner="Bootstrapping 95% confidence intervals (200 resamples)...")
def ci_table(group_col: str, cutoff: float) -> pd.DataFrame:
    report = audit(load_scores(), [group_col], cutoff, n_boot=200, seed=0)
    m = report.metrics
    keep = m[m["metric"].isin(["auc", "flag_rate", "fnr", "fpr", "ppv"])]
    keep = keep.assign(
        cell=[fmt_ci(r.metric, r.value, r.ci_low, r.ci_high) for r in keep.itertuples(index=False)]
    )
    table = keep.pivot_table(index="group", columns="metric", values="cell", aggfunc="first")
    table = table.reindex(ordered(table.index))[["auc", "flag_rate", "fnr", "fpr", "ppv"]]
    table.columns = ["AUC", "Flagged", "Missed (FNR)", "False alarms (FPR)", "PPV"]
    return table


def fmt_ci(metric, value, low, high) -> str:
    if pd.isna(value):
        return "n/a"
    if metric == "auc":
        body = f"{value:.3f}"
        ci = f" ({low:.3f}–{high:.3f})" if pd.notna(low) else ""
    else:
        body = f"{100 * value:.1f}%"
        ci = f" ({100 * low:.1f}–{100 * high:.1f}%)" if pd.notna(low) else ""
    return body + ci


def sweep_chart(data: pd.DataFrame, metric: str, label: str, share: float) -> Figure:
    fig = Figure(figsize=(8, 4.2), dpi=110, facecolor="white")
    ax = fig.subplots()
    for i, group in enumerate(ordered(data["group"].unique())):
        part = data[data["group"] == group].sort_values("flag_share")
        ax.plot(
            100 * part["flag_share"],
            100 * part[metric],
            color=SERIES[i % len(SERIES)],
            linewidth=2,
            label=group,
        )
    ax.axvline(100 * share, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(
        f"current: flag top {100 * share:.0f}%",
        (100 * share, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8,
        color="#52514e",
    )
    ax.set_xlabel("share of all patients flagged (%)", fontsize=9, color="#52514e")
    ax.set_ylabel(f"{label.split(' (')[0]} (%)", fontsize=9, color="#52514e")
    ax.set_ylim(0, 100)
    ax.grid(color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    fig.tight_layout()
    return fig


def percent_table(table: pd.DataFrame) -> pd.DataFrame:
    shown = table.copy()
    for col in ["Flagged", "Missed (FNR)", "False alarms (FPR)", "PPV"]:
        shown[col] = shown[col].map(lambda v: "n/a" if pd.isna(v) else f"{100 * v:.1f}%")
    shown["AUC"] = shown["AUC"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    for col in ["Patients", "Readmitted"]:
        shown[col] = shown[col].map(lambda v: f"{v:,}")
    return shown


# ---------------------------------------------------------------- page

st.set_page_config(page_title="EquityPanel demo", layout="wide")
st.title("EquityPanel demo")
st.caption(
    f"Measure how a clinical risk model performs for different patient groups. "
    f"[Code]({REPO}) · [Full audit report]({REPORT})"
)
st.warning(INTENDED_USE)

audit_tab, sweep_tab, egfr_tab, summary_tab, about_tab = st.tabs(
    [
        "Readmission audit",
        "Threshold explorer",
        "eGFR (synthetic)",
        "AI summary + number check",
        "About",
    ]
)

with audit_tab:
    st.markdown(
        "A baseline model predicts 30-day readmission for 20,997 held-out patients from the "
        "public UCI diabetes data. Race and sex are **not** model inputs. A program flags the "
        "highest-risk patients for follow-up; choose how many it can handle and see who is missed."
    )
    c1, c2 = st.columns(2)
    group_label = c1.selectbox("Compare groups by", list(GROUP_COLS), key="audit_group")
    share = c2.slider("Flag the top ... % of patients", 1, 50, 10, key="audit_share") / 100
    col = GROUP_COLS[group_label]
    table, cutoff = point_table(col, share)
    st.caption(
        f"Patients with a risk score of {cutoff:.3f} or more are flagged "
        f"({100 * share:.0f}% of all test patients). AUC doesn't depend on the cutoff."
    )
    st.dataframe(percent_table(table), hide_index=True, width="stretch")
    st.caption(
        "Small group: fewer than 30 readmitted or not-readmitted patients, so its numbers are "
        "uncertain. Very small: fewer than 11, which could identify people in real data."
    )
    if st.button("Add 95% confidence intervals (stratified bootstrap)", key="ci_button"):
        st.dataframe(ci_table(col, cutoff), width="stretch")
        st.caption("Estimate (95% CI). 200 resamples within each group and outcome, seed 0.")

with sweep_tab:
    st.markdown(
        "The cutoff is a choice. Move it and watch each group's error rate. Every group faces "
        "the same cutoff, as in a real program."
    )
    c1, c2, c3 = st.columns(3)
    group_label = c1.selectbox("Compare groups by", list(GROUP_COLS), key="sweep_group")
    metric_label = c2.selectbox("Show", list(METRICS), key="sweep_metric")
    share = c3.slider("Current cutoff: flag the top ... %", 1, 50, 10, key="sweep_share") / 100
    data = sweep(GROUP_COLS[group_label])
    metric = METRICS[metric_label]
    st.pyplot(sweep_chart(data, metric, metric_label, share))
    at = data[(data["flag_share"] - share).abs() < 1e-9][["group", "n", metric]]
    at = at.assign(**{metric_label: at[metric].map(lambda v: f"{100 * v:.1f}%")})
    st.dataframe(at[["group", "n", metric_label]], hide_index=True, width="stretch")

with egfr_tab:
    st.error(
        "**Synthetic patients (Synthea). Not real-world rates.** The direction matches published "
        "studies; the counts don't estimate real prevalence. The equations are computed only to "
        "count how a cohort moves, never to estimate an individual's kidney function."
    )
    st.markdown(
        "Until 2021 the standard kidney-function formula (CKD-EPI 2009) multiplied a Black "
        "patient's result by 1.159. The race-free 2021 formula moves patients across the lines "
        "where care changes."
    )
    st.image(str(DATA / "egfr_headline.png"), width="stretch")
    egfr = pd.DataFrame(load_json("egfr_summary.json"))
    rows = egfr[egfr["group_col"].isin(["all", "race_term"])]
    st.dataframe(
        pd.DataFrame(
            {
                "Group": rows["group"].replace({"All": "All patients"}),
                "Patients": rows["n"],
                "Worse stage": rows["n_lower"],
                "Better stage": rows["n_higher"],
                "eGFR ≤ 20 (2009 → 2021)": [
                    f"{a:,} → {b:,}"
                    for a, b in zip(
                        rows["waitlist_le20_2009"], rows["waitlist_le20_2021"], strict=True
                    )
                ],
            }
        ),
        hide_index=True,
        width="stretch",
    )

with summary_tab:
    summary = load_json("uci_summary.json")
    facts = load_json("uci_facts.json")["facts"]
    st.success(
        "**AI-generated summary.** An AI model wrote these findings from the report's aggregate "
        "numbers only; it saw no individual patient records. Software confirmed that each number "
        "it cites matches a computed value. The wording, comparisons and conclusions were not "
        "checked and may be wrong or misleading, and no person has reviewed this summary. Rely on "
        "the tables, and have a qualified person review before acting on anything here."
    )
    for finding in summary["findings"]:
        st.markdown(f"- {finding['text']}")
    earlier = [r for r in summary["rejected"] if r.get("attempt", 1) < summary["attempts"]]
    if earlier:
        with st.expander(f"The number check caught {len(earlier)} finding(s) in an earlier draft"):
            for r in earlier:
                st.markdown(f"- Attempt {r['attempt']}: *{r['reason']}*: “{r['text']}”")

    st.subheader("Try the number check yourself")
    st.markdown(
        "Every number in a finding must match one of the facts it cites (as written or as a "
        "percentage, at the precision shown). This runs EquityPanel's real check, locally, with "
        "no AI and no API call."
    )
    examples = {
        "A correct finding": (
            "Black patients' false negative rate is 79.6% (95% CI 75.2% to 83.7%).",
            ["race.Black.fnr"],
        ),
        "An invented number": ("Black patients are missed 91% of the time.", ["race.Black.fnr"]),
        "Wrong rounding": ("Black patients' false negative rate is 79.5%.", ["race.Black.fnr"]),
        "A number the AI computed itself": (
            "29 more Black patients reach eGFR 20 or below.",
            ["egfr.Black.waitlist_le20_2009", "egfr.Black.waitlist_le20_2021"],
        ),
    }
    choice = st.radio("Start from an example", list(examples), horizontal=True, key="example")
    text, ids = examples[choice]
    sentence = st.text_area("Finding", value=text, key=f"text_{choice}")
    cited = st.multiselect("Facts it cites", sorted(facts), default=ids, key=f"ids_{choice}")
    with st.expander("Values of the cited facts"):
        st.json({fid: facts[fid] for fid in cited})
    reason = check_finding(sentence, cited, facts)
    if reason is None:
        st.success("Passes: every number matches a cited fact.")
    else:
        st.error(f"Rejected: {reason}.")

with about_tab:
    st.markdown(
        f"""
**What this is.** An interactive view of [EquityPanel]({REPO}), an open-source Python tool that
measures differences in a clinical risk model's performance between groups, with confidence
intervals and small-group flags, plus a CKD-EPI 2009-vs-2021 eGFR comparison and an optional
AI-written summary whose numbers are matched to the computed results. The
[full audit report]({REPORT}) has every table and chart.

**Data and privacy.** This demo uses only bundled public (UCI) and synthetic (Synthea) data. It
accepts no uploads and makes no AI calls. Audit real patient data only locally, inside your
organization, under its data-governance rules.

**Limits.** One dataset (1999-2008 US hospitals) and one baseline model. Readmission labels can
carry their own bias. Cutoffs here come from the test patients' scores, so they differ slightly
from the full report, where the cutoff was set on training patients. A small or non-significant
gap does not show a model is fair.
"""
    )
    st.caption(CREDITS)

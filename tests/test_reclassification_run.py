"""Tests for the eGFR headline chart and the end-to-end run (plot.py, run.py)."""

import json
from pathlib import Path

import pandas as pd
import pytest
from matplotlib.figure import Figure

from equitypanel.reclassification.plot import SYNTHETIC_LABEL, headline_chart
from equitypanel.reclassification.reclassify import reclassification_summary, reclassify
from equitypanel.reclassification.run import OUTPUT_FILES, run_reclassification
from test_reclassify import make_lab

REAL_SYNTHEA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthea"


@pytest.fixture(scope="module")
def summary():
    return reclassification_summary(reclassify(make_lab()))


def all_text(fig: Figure) -> str:
    texts = [t.get_text() for t in fig.texts]
    if fig._suptitle is not None:
        texts.append(fig._suptitle.get_text())
    for ax in fig.axes:
        texts += [ax.get_title(loc="left"), ax.get_ylabel()]
        texts += [t.get_text() for t in ax.texts]
        texts += [t.get_text() for t in ax.get_xticklabels()]
    return "\n".join(texts)


# ---------------------------------------------------------------- headline_chart


def test_chart_is_a_figure_with_two_panels(summary):
    fig = headline_chart(summary)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_chart_says_synthetic(summary):
    assert SYNTHETIC_LABEL == "Synthetic patients (Synthea). Not real-world rates."
    assert SYNTHETIC_LABEL in all_text(headline_chart(summary))


def test_chart_panels_and_counts(summary):
    fig = headline_chart(summary)
    text = all_text(fig)
    assert "Black patients (n = 3)" in text
    assert "non-Black patients (n = 3)" in text
    assert "eGFR ≤ 20" in text
    # 3 lines x 2 years = 6 bars per panel, each with a count label
    for ax in fig.axes:
        assert len(ax.patches) == 6
        assert len(ax.texts) == 6


def test_chart_bar_heights_are_percentages(summary):
    fig = headline_chart(summary)
    black = fig.axes[0]
    # Black: ckd 2/3 -> 3/3, referral 1/3 -> 1/3, waitlist 0 -> 1/3; 2009 bars come first.
    heights = [round(p.get_height(), 4) for p in black.patches]
    assert heights == [66.6667, 33.3333, 0.0, 100.0, 33.3333, 33.3333]


def test_chart_needs_race_term_rows(summary):
    with pytest.raises(ValueError, match="race_term"):
        headline_chart(summary[summary["group_col"] != "race_term"])


# ---------------------------------------------------------------- run_reclassification


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    lab = make_lab(creatinine_mg_dl=[3.5, 1.4, 1.35, 3.3, 0.9, 25.0])  # p6 is implausible
    out = tmp_path_factory.mktemp("egfr") / "nested" / "out"  # does not exist yet
    return lab, out, run_reclassification(lab, out)


def test_run_writes_all_files(run):
    _, out, _ = run
    for name in OUTPUT_FILES:
        assert (out / name).stat().st_size > 0
    assert (out / "egfr_headline.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_headline_json(run):
    _, out, headline = run
    assert json.loads((out / "egfr_headline.json").read_text()) == headline
    assert headline["data_label"] == SYNTHETIC_LABEL
    assert headline["n_input"] == 6
    assert headline["n_dropped_implausible"] == 1
    assert headline["n_patients"] == 5
    assert headline["creatinine_range_mg_dl"] == [0.2, 20.0]
    assert headline["by_race_term"]["Black"]["n"] == 2
    assert headline["by_race_term"]["Black"]["waitlist_le20_newly_below"] == 1


def test_run_csvs(run):
    _, out, _ = run
    patients = pd.read_csv(out / "egfr_patients.csv")
    assert list(patients["patient_id"]) == ["p1", "p2", "p3", "p4", "p5"]
    summary = pd.read_csv(out / "egfr_summary.csv")
    assert summary.iloc[0]["n"] == 5
    crosstab = pd.read_csv(out / "egfr_crosstab.csv", index_col=0)
    assert crosstab.to_numpy().sum() == 5


def test_run_writes_nothing_on_failure(tmp_path):
    bad = make_lab(sex=["Unknown"] * 6)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        run_reclassification(bad, out)
    assert not out.exists()


@pytest.mark.real_data
@pytest.mark.skipif(
    not REAL_SYNTHEA_DIR.exists(),
    reason="data/raw/synthea/ (10k Synthea CSV export) not downloaded",
)
def test_real_synthea_run(tmp_path):
    from equitypanel.data.synthea import load_synthea

    headline = run_reclassification(load_synthea(REAL_SYNTHEA_DIR), tmp_path)
    assert headline["n_input"] > 5000
    assert headline["n_dropped_implausible"] < 0.01 * headline["n_input"]
    black = headline["by_race_term"]["Black"]
    # Removing the 1.159 multiplier can only lower or keep a Black patient's stage...
    assert black["n_higher"] == 0
    # ...so nobody Black leaves the waitlist range, and some enter it.
    assert black["waitlist_le20_newly_above"] == 0
    assert black["waitlist_le20_newly_below"] > 0

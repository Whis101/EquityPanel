"""Smoke tests for the Streamlit demo (demo/app.py), run headlessly with Streamlit's AppTest."""

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "demo" / "app.py"
DATA = ROOT / "demo" / "data"


@pytest.fixture
def app():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_bundled_data_is_public_scores_only():
    scores = pd.read_csv(DATA / "uci_test_scores.csv")
    assert list(scores.columns) == ["y_true", "y_score", "race", "sex", "age_band"]
    assert len(scores) == 20997
    for name in ("egfr_summary.json", "uci_facts.json", "uci_summary.json", "egfr_headline.png"):
        assert (DATA / name).exists()


def test_app_runs_and_shows_disclaimer_and_tabs(app):
    assert "Research and education demo, not a clinical tool" in app.warning[0].value
    assert [t.label for t in app.tabs] == [
        "Readmission audit",
        "Threshold explorer",
        "eGFR (synthetic)",
        "AI summary + number check",
        "About",
    ]


def test_audit_table_follows_group_and_cutoff(app):
    table = app.dataframe[0].value
    assert list(table["Group"]) == ["Asian", "Black", "Hispanic", "Other", "Unknown", "White"]
    assert table.loc[table["Group"] == "Other", "Note"].item() == "small group"
    app.selectbox(key="audit_group").set_value("Age band").run()
    assert list(app.dataframe[0].value["Group"]) == ["<50", "50-59", "60-69", "70-79", "80+"]
    flagged_10 = app.dataframe[0].value["Flagged"].tolist()
    app.slider(key="audit_share").set_value(30).run()
    assert app.dataframe[0].value["Flagged"].tolist() != flagged_10


def test_number_check_rejects_the_invented_example(app):
    app.radio(key="example").set_value("An invented number").run()
    assert not app.exception
    assert any("Rejected: 91% does not match any cited fact" in e.value for e in app.error)
    app.radio(key="example").set_value("A correct finding").run()
    assert any("Passes" in s.value for s in app.success)


def test_number_check_rejects_self_computed_numbers(app):
    app.radio(key="example").set_value("A number the AI computed itself").run()
    assert any("Rejected: 29 does not match any cited fact" in e.value for e in app.error)


def test_egfr_tab_is_labelled_synthetic(app):
    assert any("Synthetic patients (Synthea)" in e.value for e in app.error)


def test_confidence_intervals_button(app):
    app.button(key="ci_button").click().run()
    assert not app.exception
    ci = app.dataframe[1].value
    assert "AUC" in ci.columns and "(" in str(ci.iloc[0, 0])

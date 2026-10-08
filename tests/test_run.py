"""Tests for equitypanel.model.run: the end-to-end UCI audit (split, fit, score, audit, write)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equitypanel.model.baseline import (
    capacity_threshold,
    fit_baseline,
    predict_risk,
    split_cohort,
)
from equitypanel.model.run import OUTPUT_FILES, run_audit

REAL_UCI_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetic_data.csv"
SUMMARY_KEYS = {
    "n_train",
    "n_test",
    "prevalence_test",
    "auc_test",
    "threshold",
    "flag_share_test",
    "top_weights",
}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from conftest import make_cohort

    cohort = make_cohort(n=3000, seed=2)
    out = tmp_path_factory.mktemp("run") / "nested" / "out"  # does not exist yet
    summary = run_audit(cohort, out, n_boot=30, seed=0)
    return cohort, out, summary


def test_writes_all_output_files(run):
    _, out, _ = run
    assert OUTPUT_FILES == (
        "uci_scores.csv",
        "uci_audit_metrics.csv",
        "uci_audit_calibration.csv",
        "uci_model.json",
    )
    for name in OUTPUT_FILES:
        assert (out / name).is_file(), name


def test_summary_contents(run):
    cohort, out, summary = run
    assert set(summary) == SUMMARY_KEYS
    assert summary["n_train"] + summary["n_test"] == len(cohort)
    assert summary["n_test"] == 900
    assert 0.5 < summary["auc_test"] <= 1
    assert 0 < summary["threshold"] < 1
    assert 0.05 < summary["flag_share_test"] < 0.15
    assert json.loads((out / "uci_model.json").read_text()) == summary


def test_top_weights_are_listed(run):
    _, _, summary = run
    top = summary["top_weights"]
    assert 1 <= len(top) <= 15
    assert set(top[0]) == {"feature", "weight"}


def test_threshold_comes_from_training_scores_only(run):
    cohort, _, summary = run
    train, _ = split_cohort(cohort, seed=0)
    model = fit_baseline(train)
    expected = capacity_threshold(predict_risk(model, train), flag_share=0.10)
    assert summary["threshold"] == pytest.approx(expected)


def test_scores_file_holds_the_test_set_only(run):
    cohort, out, summary = run
    scores = pd.read_csv(out / "uci_scores.csv")
    assert list(scores.columns) == ["patient_id", "y_true", "y_score", "race", "sex", "age_band"]
    assert len(scores) == summary["n_test"]
    _, test = split_cohort(cohort, seed=0)
    assert set(scores["patient_id"]) == set(test["patient_id"])


def test_metrics_cover_all_three_group_columns(run):
    _, out, _ = run
    m = pd.read_csv(out / "uci_audit_metrics.csv")
    assert list(m["group_col"].unique()) == ["race", "sex", "age_band"]
    n_rows = m[m["metric"] == "n"]
    assert n_rows.groupby("group_col")["value"].sum().eq(900).all()


def test_calibration_file(run):
    _, out, _ = run
    cal = pd.read_csv(out / "uci_audit_calibration.csv")
    assert list(cal.columns) == ["group_col", "group", "bin", "mean_pred", "obs_rate", "n"]


def test_rerun_is_reproducible(run, tmp_path):
    cohort, out, summary = run
    again = run_audit(cohort, tmp_path, n_boot=30, seed=0)
    assert again == summary
    a = pd.read_csv(out / "uci_audit_metrics.csv")
    b = pd.read_csv(tmp_path / "uci_audit_metrics.csv")
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("flag_share", [0, 1])
def test_bad_flag_share_raises_before_writing(tmp_path, flag_share):
    from conftest import make_cohort

    with pytest.raises(ValueError, match="flag_share"):
        run_audit(make_cohort(n=200), tmp_path / "x", n_boot=5, flag_share=flag_share)
    assert not (tmp_path / "x").exists()


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_UCI_PATH.exists(), reason="data/raw/diabetic_data.csv not downloaded")
def test_real_uci_end_to_end(tmp_path):
    from equitypanel.data.uci import load_uci

    summary = run_audit(load_uci(REAL_UCI_PATH), tmp_path, n_boot=20, seed=0)
    assert summary["n_test"] == pytest.approx(0.3 * 69_990, abs=1)
    assert 0.60 < summary["auc_test"] < 0.75  # published models: about 0.64-0.68
    assert 0.08 < summary["flag_share_test"] < 0.12
    assert np.isfinite(summary["threshold"])

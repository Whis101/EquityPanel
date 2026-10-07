"""Tests for equitypanel.metrics.audit: the per-group report card for an audit frame."""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from equitypanel.data.schema import SchemaError
from equitypanel.metrics.audit import AuditReport, audit
from equitypanel.metrics.calibration import calibration_curve
from equitypanel.metrics.core import auc, brier, o_e, threshold_rates

METRIC_COLUMNS = [
    "group_col",
    "group",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n",
    "small_group",
    "n_undefined",
]
CALIBRATION_COLUMNS = ["group_col", "group", "bin", "mean_pred", "obs_rate", "n"]
BASE_METRICS = [
    "n",
    "n_pos",
    "prevalence",
    "auc",
    "brier",
    "o_e",
    "flag_rate",
    "fpr",
    "fnr",
    "ppv",
]
GAP_METRICS = [f"{m}_gap" for m in ["auc", "brier", "o_e", "flag_rate", "fpr", "fnr", "ppv"]]
T = 0.3
B = 200


def make_frame(seed=0):
    """300 patients: race A (200), B (80), C (20, small); sex F/M, independent of race."""
    rng = np.random.default_rng(seed)
    race = np.array(["A"] * 200 + ["B"] * 80 + ["C"] * 20)
    sex = rng.choice(["F", "M"], size=300)
    s = rng.beta(2, 5, size=300)
    y = rng.binomial(1, s)
    return pd.DataFrame({"y_true": y, "y_score": s, "race": race, "sex": sex})


@pytest.fixture(scope="module")
def frame():
    return make_frame()


@pytest.fixture(scope="module")
def report(frame):
    return audit(frame, ["race", "sex"], T, n_boot=B, seed=0)


def rows(report, col, group):
    m = report.metrics
    return m[(m["group_col"] == col) & (m["group"] == group)].set_index("metric")


# ---------------------------------------------------------------- shape of the output


def test_returns_frozen_report_with_two_tables(report):
    assert isinstance(report, AuditReport)
    assert isinstance(report.metrics, pd.DataFrame)
    assert isinstance(report.calibration, pd.DataFrame)
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.metrics = None


def test_metrics_columns_and_dtypes(report):
    m = report.metrics
    assert list(m.columns) == METRIC_COLUMNS
    for col in ["value", "ci_low", "ci_high"]:
        assert pd.api.types.is_float_dtype(m[col])
    assert pd.api.types.is_integer_dtype(m["n"])
    assert pd.api.types.is_integer_dtype(m["n_undefined"])
    assert pd.api.types.is_bool_dtype(m["small_group"])
    assert m.index.equals(pd.RangeIndex(len(m)))


def test_row_order_columns_then_sorted_groups_then_metrics(report, frame):
    m = report.metrics
    keys = list(dict.fromkeys(zip(m["group_col"], m["group"], strict=True)))
    assert keys == [("race", "A"), ("race", "B"), ("race", "C"), ("sex", "F"), ("sex", "M")]
    # Reference group (largest) has no gap rows; the others get all 7 gaps.
    assert rows(report, "race", "A").index.tolist() == BASE_METRICS
    assert rows(report, "race", "B").index.tolist() == BASE_METRICS + GAP_METRICS
    sex_ref = frame["sex"].value_counts().idxmax()
    sex_other = "M" if sex_ref == "F" else "F"
    assert rows(report, "sex", sex_ref).index.tolist() == BASE_METRICS
    assert rows(report, "sex", sex_other).index.tolist() == BASE_METRICS + GAP_METRICS


def test_prevalence_has_no_gap(report):
    assert not report.metrics["metric"].eq("prevalence_gap").any()


# ---------------------------------------------------------------- values


def test_values_match_core_functions_on_each_group(report, frame):
    for group in ["A", "B", "C"]:
        sub = frame[frame["race"] == group]
        y, s = sub["y_true"].to_numpy(), sub["y_score"].to_numpy()
        r = rows(report, "race", group)["value"]
        assert r["n"] == len(sub)
        assert r["n_pos"] == y.sum()
        assert r["prevalence"] == pytest.approx(y.mean())
        assert r["auc"] == pytest.approx(auc(y, s), nan_ok=True)
        assert r["brier"] == pytest.approx(brier(y, s))
        assert r["o_e"] == pytest.approx(o_e(y, s))
        for name, value in threshold_rates(y, s, T).items():
            assert r[name] == pytest.approx(value, nan_ok=True)


def test_gap_is_group_minus_reference(report):
    a = rows(report, "race", "A")["value"]
    b = rows(report, "race", "B")["value"]
    for m in ["auc", "brier", "o_e", "flag_rate", "fpr", "fnr", "ppv"]:
        assert b[f"{m}_gap"] == pytest.approx(b[m] - a[m], nan_ok=True)


def test_n_and_small_group_on_every_row(report):
    c = rows(report, "race", "C")
    assert (c["n"] == 20).all()
    assert c["small_group"].all()
    assert not rows(report, "race", "A")["small_group"].any()


def test_min_group_size_moves_the_small_group_flag(frame):
    rep = audit(frame, ["race"], T, n_boot=20, seed=0, min_group_size=10)
    assert not rep.metrics["small_group"].any()
    rep = audit(frame, ["race"], T, n_boot=20, seed=0, min_group_size=81)
    assert rows(rep, "race", "B")["small_group"].all()


def test_group_values_are_strings():
    df = make_frame()
    df["race"] = df["race"].map({"A": 1, "B": 2, "C": 3})
    rep = audit(df, ["race"], T, n_boot=20, seed=0)
    assert set(rep.metrics["group"]) == {"1", "2", "3"}
    assert set(rep.calibration["group"]) == {"1", "2", "3"}


# ---------------------------------------------------------------- confidence intervals


def test_counts_have_no_ci_and_rates_do(report):
    a = rows(report, "race", "A")
    assert a.loc[["n", "n_pos"], ["ci_low", "ci_high"]].isna().all().all()
    for m in ["prevalence", "auc", "brier", "o_e", "fnr"]:
        assert a.loc[m, "ci_low"] <= a.loc[m, "value"] <= a.loc[m, "ci_high"]


def test_ci_low_never_above_ci_high(report):
    m = report.metrics.dropna(subset=["ci_low", "ci_high"])
    assert (m["ci_low"] <= m["ci_high"]).all()


def test_smaller_group_gets_wider_ci(report):
    def width(group):
        r = rows(report, "race", group).loc["prevalence"]
        return r["ci_high"] - r["ci_low"]

    assert width("C") > width("B") > width("A")


def test_same_seed_same_report(frame, report):
    again = audit(frame, ["race", "sex"], T, n_boot=B, seed=0)
    pd.testing.assert_frame_equal(report.metrics, again.metrics)


def test_different_seed_changes_cis_not_values(frame, report):
    other = audit(frame, ["race", "sex"], T, n_boot=B, seed=1)
    pd.testing.assert_series_equal(report.metrics["value"], other.metrics["value"])
    assert not report.metrics["ci_low"].equals(other.metrics["ci_low"])


def test_adding_a_group_column_does_not_change_other_cis(frame, report):
    race_only = audit(frame, ["race"], T, n_boot=B, seed=0).metrics
    race_rows = report.metrics[report.metrics["group_col"] == "race"].reset_index(drop=True)
    pd.testing.assert_frame_equal(race_only, race_rows)


def test_one_class_group_gives_nan_not_error():
    df = make_frame()
    df.loc[df["race"] == "C", "y_true"] = 0  # no readmissions in C
    rep = audit(df, ["race"], T, n_boot=50, seed=0)
    c = rows(rep, "race", "C")
    for m in ["auc", "fnr", "auc_gap", "fnr_gap"]:
        assert math.isnan(c.loc[m, "value"])
        assert math.isnan(c.loc[m, "ci_low"]) and math.isnan(c.loc[m, "ci_high"])
        assert c.loc[m, "n_undefined"] == 50
    assert c.loc["n_pos", "value"] == 0
    assert c.loc["prevalence", "value"] == 0
    assert c.loc["n", "n_undefined"] == 0


def test_n_undefined_counts_skipped_replicates():
    # A group of 2 with one readmission: a replicate has no AUC whenever it draws the
    # same patient twice (about half the time), so some but not all replicates are skipped.
    df = make_frame()
    df = pd.concat(
        [df, pd.DataFrame({"y_true": [0, 1], "y_score": [0.2, 0.6], "race": "D", "sex": "F"})],
        ignore_index=True,
    )
    rep = audit(df, ["race"], T, n_boot=100, seed=0)
    d = rows(rep, "race", "D")
    assert 0 < d.loc["auc", "n_undefined"] < 100
    assert d.loc["auc", "value"] == pytest.approx(1.0)
    assert rows(rep, "race", "A").loc["auc", "n_undefined"] == 0


# ---------------------------------------------------------------- reference group


def test_reference_tie_goes_to_first_in_sorted_order():
    df = make_frame()
    df["grp"] = ["y", "x"] * 150  # 150 each
    rep = audit(df, ["grp"], T, n_boot=20, seed=0)
    assert "auc_gap" not in rows(rep, "grp", "x").index
    assert "auc_gap" in rows(rep, "grp", "y").index


def test_reference_can_be_overridden(frame):
    rep = audit(frame, ["race", "sex"], T, n_boot=20, seed=0, reference={"race": "B"})
    assert "auc_gap" not in rows(rep, "race", "B").index
    assert "auc_gap" in rows(rep, "race", "A").index
    a, b = rows(rep, "race", "A")["value"], rows(rep, "race", "B")["value"]
    assert a["fnr_gap"] == pytest.approx(a["fnr"] - b["fnr"])


def test_reference_matches_string_form_of_group():
    df = make_frame()
    df["race"] = df["race"].map({"A": 1, "B": 2, "C": 3})
    rep = audit(df, ["race"], T, n_boot=20, seed=0, reference={"race": 2})
    assert "auc_gap" not in rows(rep, "race", "2").index


@pytest.mark.parametrize("reference", [{"race": "Z"}, {"age_band": "A"}])
def test_reference_must_exist(frame, reference):
    with pytest.raises(ValueError, match="reference"):
        audit(frame, ["race"], T, n_boot=20, seed=0, reference=reference)


# ---------------------------------------------------------------- calibration table


def test_calibration_table(report, frame):
    cal = report.calibration
    assert list(cal.columns) == CALIBRATION_COLUMNS
    assert pd.api.types.is_integer_dtype(cal["bin"])
    assert pd.api.types.is_integer_dtype(cal["n"])
    assert cal.index.equals(pd.RangeIndex(len(cal)))
    for col in ["race", "sex"]:
        sub = cal[cal["group_col"] == col]
        assert sub["n"].sum() == len(frame)
    b = cal[(cal["group_col"] == "race") & (cal["group"] == "B")].reset_index(drop=True)
    sub = frame[frame["race"] == "B"]
    expected = calibration_curve(sub["y_true"], sub["y_score"], n_bins=10)
    pd.testing.assert_frame_equal(b[expected.columns.tolist()], expected)


def test_n_bins_is_passed_to_calibration(frame):
    rep = audit(frame, ["race"], T, n_boot=20, seed=0, n_bins=3)
    assert rep.calibration.groupby("group")["bin"].max().max() <= 2


# ---------------------------------------------------------------- safeguards


def test_does_not_modify_input(frame):
    before = frame.copy()
    audit(frame, ["race"], T, n_boot=20, seed=0)
    pd.testing.assert_frame_equal(frame, before)


def test_invalid_frame_raises_schema_error(frame):
    bad = frame.assign(y_score=frame["y_score"] + 1)
    with pytest.raises(SchemaError):
        audit(bad, ["race"], T, n_boot=20, seed=0)


def test_missing_group_column_raises_schema_error(frame):
    with pytest.raises(SchemaError):
        audit(frame, ["age_band"], T, n_boot=20, seed=0)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, None, "0.3", True])
def test_bad_threshold_raises(frame, threshold):
    with pytest.raises(ValueError, match="threshold"):
        audit(frame, ["race"], threshold, n_boot=20, seed=0)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_boot": 0}, "n_boot"),
        ({"ci": 1.0}, "ci"),
        ({"min_group_size": 0}, "min_group_size"),
        ({"n_bins": 0}, "n_bins"),
    ],
)
def test_bad_options_raise(frame, kwargs, match):
    with pytest.raises(ValueError, match=match):
        audit(frame, ["race"], T, **{"n_boot": 20, "seed": 0, **kwargs})


def test_options_are_keyword_only(frame):
    with pytest.raises(TypeError):
        audit(frame, ["race"], T, 20)

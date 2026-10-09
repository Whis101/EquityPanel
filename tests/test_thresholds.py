"""Tests for equitypanel.thresholds: capacity cutoffs and the per-group threshold sweep."""

import numpy as np
import pandas as pd
import pytest

from equitypanel.data.schema import SchemaError
from equitypanel.metrics.core import threshold_rates
from equitypanel.thresholds import capacity_thresholds, threshold_sweep
from equitypanel.thresholds.sweep import SWEEP_COLUMNS


def make_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.25, n)
    score = np.clip(0.25 + 0.3 * (y - 0.25) + rng.normal(0, 0.15, n), 0.001, 0.999)
    return pd.DataFrame(
        {"y_true": y, "y_score": score, "race": rng.choice(["A", "B", "C"], n, p=[0.5, 0.3, 0.2])}
    )


def test_capacity_thresholds_flag_about_the_share():
    scores = np.linspace(0, 1, 1001)
    cutoffs = capacity_thresholds(scores, [0.1, 0.5])
    assert np.mean(scores >= cutoffs[0]) == pytest.approx(0.1, abs=0.002)
    assert np.mean(scores >= cutoffs[1]) == pytest.approx(0.5, abs=0.002)


def test_capacity_thresholds_fall_as_share_rises():
    cutoffs = capacity_thresholds(make_frame()["y_score"], [0.05, 0.1, 0.2, 0.5, 1.0])
    assert np.all(np.diff(cutoffs) <= 0)


@pytest.mark.parametrize("share", [0, -0.1, 1.1])
def test_capacity_thresholds_reject_bad_shares(share):
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        capacity_thresholds([0.1, 0.2], [share])


def test_sweep_shape_and_columns():
    got = threshold_sweep(make_frame(), "race", shares=[0.1, 0.2, 0.3])
    assert list(got.columns) == list(SWEEP_COLUMNS)
    assert len(got) == 3 * 3
    assert list(got["group"].unique()) == ["A", "B", "C"]


def test_sweep_matches_threshold_rates_for_each_group():
    df = make_frame()
    got = threshold_sweep(df, "race", shares=[0.15])
    cutoff = float(capacity_thresholds(df["y_score"], [0.15])[0])
    for _, row in got.iterrows():
        part = df[df["race"] == row["group"]]
        expected = threshold_rates(part["y_true"], part["y_score"], cutoff)
        assert row["threshold"] == pytest.approx(cutoff)
        assert row["n"] == len(part)
        for key in ("flag_rate", "fpr", "fnr", "ppv"):
            assert row[key] == pytest.approx(expected[key], nan_ok=True)


def test_every_group_faces_the_same_cutoff():
    got = threshold_sweep(make_frame(), "race", shares=[0.1, 0.4])
    assert got.groupby("flag_share")["threshold"].nunique().eq(1).all()


def test_flagging_more_patients_misses_fewer():
    got = threshold_sweep(make_frame(n=2000), "race")
    for _, part in got.groupby("group"):
        fnr = part.sort_values("flag_share")["fnr"].to_numpy()
        assert np.all(np.diff(fnr) <= 1e-12)


def test_sweep_validates_the_frame():
    with pytest.raises(SchemaError):
        threshold_sweep(make_frame().drop(columns="y_score"), "race")

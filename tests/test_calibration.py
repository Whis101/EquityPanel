"""Tests for equitypanel.metrics.calibration: equal-count (quantile) calibration curves."""

import numpy as np
import pandas as pd
import pytest

from equitypanel.metrics.calibration import calibration_curve

COLUMNS = ["bin", "mean_pred", "obs_rate", "n"]


def test_columns_and_dtypes():
    curve = calibration_curve([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4], n_bins=2)
    assert list(curve.columns) == COLUMNS
    assert pd.api.types.is_integer_dtype(curve["bin"])
    assert pd.api.types.is_integer_dtype(curve["n"])
    assert pd.api.types.is_float_dtype(curve["mean_pred"])
    assert pd.api.types.is_float_dtype(curve["obs_rate"])


def test_equal_count_bins_on_distinct_scores():
    y = [0, 0, 1, 0, 1, 1, 0, 1]
    s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    curve = calibration_curve(y, s, n_bins=4)
    assert curve["bin"].tolist() == [0, 1, 2, 3]
    assert curve["n"].tolist() == [2, 2, 2, 2]
    assert curve["mean_pred"].tolist() == pytest.approx([0.15, 0.35, 0.55, 0.75])
    assert curve["obs_rate"].tolist() == pytest.approx([0.0, 0.5, 1.0, 0.5])


def test_bins_hold_similar_counts_on_skewed_scores():
    # Scores bunched near 0, like real readmission risk: quantile bins stay balanced.
    rng = np.random.default_rng(0)
    s = rng.beta(1, 9, size=1000)
    y = rng.binomial(1, s)
    curve = calibration_curve(y, s, n_bins=10)
    assert len(curve) == 10
    assert curve["n"].sum() == 1000
    assert curve["n"].between(95, 105).all()


def test_mean_pred_increases_across_bins():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=500)
    curve = calibration_curve(rng.binomial(1, s), s, n_bins=5)
    assert curve["mean_pred"].is_monotonic_increasing


def test_tied_scores_are_never_split_across_bins():
    s = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9]
    y = [0, 0, 1, 0, 1, 0, 1, 1]
    curve = calibration_curve(y, s, n_bins=4)
    assert curve["n"].sum() == len(s)
    # Each bin's scores must be a set of whole tie-groups: the 0.2s all share one bin.
    rows_with_02 = curve[(curve["mean_pred"] > 0.1) & (curve["mean_pred"] < 0.9)]
    assert (rows_with_02["n"] >= 5).all()


def test_all_scores_equal_gives_one_bin():
    curve = calibration_curve([0, 1, 1, 0], [0.3] * 4, n_bins=10)
    assert len(curve) == 1
    row = curve.iloc[0]
    assert row["bin"] == 0
    assert row["n"] == 4
    assert row["mean_pred"] == pytest.approx(0.3)
    assert row["obs_rate"] == pytest.approx(0.5)


def test_bins_are_numbered_from_zero_without_gaps():
    s = [0.1] * 6 + [0.5, 0.9]
    curve = calibration_curve([0] * 8, s, n_bins=10)
    assert curve["bin"].tolist() == list(range(len(curve)))


def test_n_bins_one_is_whole_group():
    curve = calibration_curve([0, 1, 1], [0.2, 0.4, 0.9], n_bins=1)
    assert len(curve) == 1
    assert curve.iloc[0]["obs_rate"] == pytest.approx(2 / 3)


def test_empty_input_gives_empty_frame_with_columns():
    curve = calibration_curve([], [], n_bins=10)
    assert curve.empty
    assert list(curve.columns) == COLUMNS


@pytest.mark.parametrize("n_bins", [0, -1, 2.5, "10"])
def test_bad_n_bins_raises(n_bins):
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve([0, 1], [0.2, 0.8], n_bins=n_bins)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        calibration_curve([0, 1, 1], [0.2, 0.8], n_bins=2)

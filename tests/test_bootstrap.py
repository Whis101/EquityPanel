"""Tests for equitypanel.metrics.bootstrap: stratified resampling and percentile CIs."""

import math

import numpy as np
import pytest

from equitypanel.metrics.bootstrap import percentile_ci, stratified_replicates


def mean_score(y, s):
    return {"mean_s": float(np.mean(s))}


# ---------------------------------------------------------------- percentile_ci


def test_percentile_ci_middle_95_percent():
    assert percentile_ci(np.arange(101.0)) == pytest.approx((2.5, 97.5))


def test_percentile_ci_other_level():
    assert percentile_ci(np.arange(101.0), ci=0.9) == pytest.approx((5.0, 95.0))


def test_percentile_ci_skips_undefined_replicates():
    values = np.concatenate([np.arange(101.0), [np.nan] * 10])
    assert percentile_ci(values) == pytest.approx((2.5, 97.5))


def test_percentile_ci_exactly_half_defined_still_gives_ci():
    low, high = percentile_ci([1.0, 2.0, np.nan, np.nan])
    assert low == pytest.approx(1.025)
    assert high == pytest.approx(1.975)


def test_percentile_ci_mostly_undefined_is_nan():
    # Only 1 of 3 defined: the rest would be a biased subset, so no CI.
    low, high = percentile_ci([1.0, np.nan, np.nan])
    assert math.isnan(low) and math.isnan(high)


@pytest.mark.parametrize("values", [[], [np.nan, np.nan]])
def test_percentile_ci_nothing_defined_is_nan(values):
    low, high = percentile_ci(values)
    assert math.isnan(low) and math.isnan(high)


def test_percentile_ci_returns_python_floats():
    low, high = percentile_ci([0.1, 0.2, 0.3])
    assert type(low) is float and type(high) is float


@pytest.mark.parametrize("ci", [0, 1, -0.1, 1.5, True, "0.95"])
def test_percentile_ci_rejects_bad_level(ci):
    with pytest.raises(ValueError, match="ci"):
        percentile_ci([0.1, 0.2], ci=ci)


# ---------------------------------------------------------------- stratified_replicates


def two_groups():
    # Group "a": 6 patients scored 0.1. Group "b": 3 patients scored 0.9.
    y = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0])
    s = np.array([0.1] * 6 + [0.9] * 3)
    groups = np.array(["a"] * 6 + ["b"] * 3)
    return y, s, groups


def test_replicates_shape_one_array_per_group_and_metric():
    y, s, g = two_groups()
    reps = stratified_replicates(y, s, g, mean_score, n_boot=50, seed=0)
    assert list(reps) == ["a", "b"]
    for group in reps:
        assert list(reps[group]) == ["mean_s"]
        assert reps[group]["mean_s"].shape == (50,)


def test_replicates_groups_come_out_sorted():
    y, s, g = two_groups()
    reps = stratified_replicates(y[::-1], s[::-1], g[::-1], mean_score, n_boot=5, seed=0)
    assert list(reps) == ["a", "b"]


def test_each_group_resamples_only_its_own_patients():
    y, s, g = two_groups()
    reps = stratified_replicates(y, s, g, mean_score, n_boot=100, seed=0)
    assert np.all(reps["a"]["mean_s"] == pytest.approx(0.1))
    assert np.all(reps["b"]["mean_s"] == pytest.approx(0.9))


def test_each_replicate_keeps_the_group_size():
    y, s, g = two_groups()
    reps = stratified_replicates(y, s, g, lambda y, s: {"n": float(len(y))}, n_boot=20, seed=0)
    assert np.all(reps["a"]["n"] == 6)
    assert np.all(reps["b"]["n"] == 3)


def test_resampling_is_with_replacement():
    s = np.arange(10) / 10
    y = np.zeros(10)
    g = np.array(["a"] * 10)
    reps = stratified_replicates(
        y, s, g, lambda y, s: {"distinct": float(len(np.unique(s)))}, n_boot=200, seed=0
    )
    assert reps["a"]["distinct"].max() <= 10
    assert reps["a"]["distinct"].min() < 10  # some patients drawn twice, some left out


def test_same_seed_same_replicates():
    rng = np.random.default_rng(1)
    s = rng.random(40)
    y = rng.integers(0, 2, 40)
    g = np.array(["a", "b"] * 20)
    r1 = stratified_replicates(y, s, g, mean_score, n_boot=30, seed=7)
    r2 = stratified_replicates(y, s, g, mean_score, n_boot=30, seed=7)
    r3 = stratified_replicates(y, s, g, mean_score, n_boot=30, seed=8)
    np.testing.assert_array_equal(r1["a"]["mean_s"], r2["a"]["mean_s"])
    assert not np.array_equal(r1["a"]["mean_s"], r3["a"]["mean_s"])


def test_draw_order_matches_spec():
    # Pins the documented order: one rng from the seed; for each replicate, groups in sorted
    # order, each drawing rng.integers(0, n_g, n_g). Changing it would silently change CIs.
    rng = np.random.default_rng(3)
    s = rng.random(15)
    y = rng.integers(0, 2, 15)
    g = np.array(["b", "a", "c"] * 5)
    reps = stratified_replicates(y, s, g, mean_score, n_boot=10, seed=42)

    expected = {k: [] for k in ["a", "b", "c"]}
    ref_rng = np.random.default_rng(42)
    for _ in range(10):
        for k in ["a", "b", "c"]:
            s_k = s[g == k]
            idx = ref_rng.integers(0, len(s_k), len(s_k))
            expected[k].append(s_k[idx].mean())
    for k in expected:
        np.testing.assert_allclose(reps[k]["mean_s"], expected[k])


def test_undefined_replicates_are_kept_as_nan():
    y, s, g = two_groups()
    reps = stratified_replicates(y, s, g, lambda y, s: {"m": math.nan}, n_boot=8, seed=0)
    assert reps["a"]["m"].shape == (8,)
    assert np.isnan(reps["a"]["m"]).all()


@pytest.mark.parametrize("n_boot", [0, -1, 1.5, True])
def test_rejects_bad_n_boot(n_boot):
    y, s, g = two_groups()
    with pytest.raises(ValueError, match="n_boot"):
        stratified_replicates(y, s, g, mean_score, n_boot=n_boot, seed=0)


def test_rejects_length_mismatch():
    y, s, g = two_groups()
    with pytest.raises(ValueError, match="same length"):
        stratified_replicates(y, s, g[:-1], mean_score, n_boot=5, seed=0)

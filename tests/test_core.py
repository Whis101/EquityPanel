"""Tests for equitypanel.metrics.core: per-group metrics on plain arrays."""

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from equitypanel.metrics.core import (
    auc,
    brier,
    confusion_counts,
    o_e,
    threshold_rates,
)

# ---------------------------------------------------------------- auc


def test_auc_worked_example():
    # Readmitted [0.8, 0.4], not readmitted [0.5, 0.1]: 3 of 4 pairs ranked right.
    y = [1, 1, 0, 0]
    s = [0.8, 0.4, 0.5, 0.1]
    assert auc(y, s) == pytest.approx(0.75)


def test_auc_perfect_and_reversed():
    y = [0, 0, 1, 1]
    assert auc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert auc(y, [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


def test_auc_all_scores_equal_is_half():
    assert auc([0, 1, 0, 1, 1], [0.3] * 5) == pytest.approx(0.5)


def test_auc_tie_across_classes_counts_half():
    # One pair, tied: counts as half a correct ranking.
    assert auc([1, 0], [0.4, 0.4]) == pytest.approx(0.5)


def test_auc_order_of_rows_does_not_matter():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.3, 0.3, 0.6, 0.1, 0.5])
    perm = np.array([5, 2, 0, 4, 1, 3])
    assert auc(y, s) == pytest.approx(auc(y[perm], s[perm]))


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn_on_random_data_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = 500
    y = rng.integers(0, 2, size=n)
    s = rng.integers(0, 20, size=n) / 20  # only 20 distinct values -> many ties
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


def test_auc_matches_sklearn_on_continuous_scores():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=2000)
    s = np.clip(0.3 * y + rng.normal(0.3, 0.2, size=2000), 0, 1)
    assert auc(y, s) == pytest.approx(roc_auc_score(y, s))


@pytest.mark.parametrize("y", [[1, 1, 1], [0, 0, 0], []])
def test_auc_undefined_without_both_classes_is_nan(y):
    s = [0.5] * len(y)
    assert math.isnan(auc(y, s))


def test_auc_returns_python_float():
    assert isinstance(auc([0, 1], [0.2, 0.7]), float)


# ---------------------------------------------------------------- brier


def test_brier_worked_example():
    # (0.2-0)^2 = 0.04 and (0.2-1)^2 = 0.64 -> mean 0.34
    assert brier([0, 1], [0.2, 0.2]) == pytest.approx(0.34)


def test_brier_perfect_is_zero():
    assert brier([0, 1, 1], [0.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_brier_empty_is_nan():
    assert math.isnan(brier([], []))


# ---------------------------------------------------------------- o_e


def test_o_e_worked_example():
    # 2 observed readmissions, 0.5 * 4 = 2 expected -> 1.0
    assert o_e([1, 1, 0, 0], [0.5] * 4) == pytest.approx(1.0)


def test_o_e_above_one_means_underestimated():
    # 2 observed, 0.25 * 4 = 1 expected -> 2.0
    assert o_e([1, 1, 0, 0], [0.25] * 4) == pytest.approx(2.0)


@pytest.mark.parametrize("y, s", [([0, 1], [0.0, 0.0]), ([], [])])
def test_o_e_undefined_when_nothing_expected(y, s):
    assert math.isnan(o_e(y, s))


# ---------------------------------------------------------------- confusion_counts


def test_confusion_counts_basic():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.1, 0.8, 0.2, 0.6, 0.4]
    # threshold 0.5 -> flagged: 0.9 (TP), 0.8 (FP), 0.6 (TP)
    assert confusion_counts(y, s, 0.5) == (2, 1, 1, 2)


def test_confusion_counts_score_equal_to_threshold_is_flagged():
    assert confusion_counts([1, 0], [0.3, 0.3], 0.3) == (1, 1, 0, 0)


def test_confusion_counts_threshold_zero_flags_everyone():
    assert confusion_counts([1, 0, 0], [0.0, 0.0, 0.5], 0.0) == (1, 2, 0, 0)


def test_confusion_counts_threshold_one_flags_only_exact_ones():
    assert confusion_counts([1, 1, 0], [1.0, 0.99, 1.0], 1.0) == (1, 1, 1, 0)


def test_confusion_counts_empty_is_all_zero():
    assert confusion_counts([], [], 0.5) == (0, 0, 0, 0)


def test_confusion_counts_returns_python_ints():
    assert all(type(c) is int for c in confusion_counts([1, 0], [0.9, 0.1], 0.5))


# ---------------------------------------------------------------- threshold_rates


def test_threshold_rates_values():
    y = [1, 1, 0, 0, 1, 0]
    s = [0.9, 0.1, 0.8, 0.2, 0.6, 0.4]
    # tp=2, fp=1, fn=1, tn=2
    rates = threshold_rates(y, s, 0.5)
    assert set(rates) == {"flag_rate", "fpr", "fnr", "ppv"}
    assert rates["flag_rate"] == pytest.approx(3 / 6)
    assert rates["fpr"] == pytest.approx(1 / 3)
    assert rates["fnr"] == pytest.approx(1 / 3)
    assert rates["ppv"] == pytest.approx(2 / 3)


def test_threshold_rates_no_positives_fnr_nan():
    rates = threshold_rates([0, 0], [0.9, 0.1], 0.5)
    assert math.isnan(rates["fnr"])
    assert rates["fpr"] == pytest.approx(0.5)


def test_threshold_rates_no_negatives_fpr_nan():
    rates = threshold_rates([1, 1], [0.9, 0.1], 0.5)
    assert math.isnan(rates["fpr"])
    assert rates["fnr"] == pytest.approx(0.5)


def test_threshold_rates_nobody_flagged_ppv_nan():
    rates = threshold_rates([1, 0], [0.1, 0.2], 0.5)
    assert math.isnan(rates["ppv"])
    assert rates["flag_rate"] == pytest.approx(0.0)
    assert rates["fnr"] == pytest.approx(1.0)


def test_threshold_rates_empty_all_nan():
    rates = threshold_rates([], [], 0.5)
    assert all(math.isnan(v) for v in rates.values())


# ---------------------------------------------------------------- input checks


@pytest.mark.parametrize("fn", [auc, brier, o_e])
def test_length_mismatch_raises(fn):
    with pytest.raises(ValueError, match="same length"):
        fn([0, 1, 1], [0.2, 0.3])


def test_length_mismatch_raises_for_threshold_functions():
    with pytest.raises(ValueError, match="same length"):
        confusion_counts([0, 1], [0.2], 0.5)
    with pytest.raises(ValueError, match="same length"):
        threshold_rates([0, 1], [0.2], 0.5)


@pytest.mark.parametrize("fn", [auc, brier, o_e])
def test_two_dimensional_input_raises(fn):
    with pytest.raises(ValueError, match="1-D"):
        fn([[0, 1]], [[0.2, 0.3]])


def test_accepts_pandas_series():
    pd = pytest.importorskip("pandas")
    y = pd.Series([1, 1, 0, 0], index=[10, 20, 30, 40])
    s = pd.Series([0.8, 0.4, 0.5, 0.1], index=[10, 20, 30, 40])
    assert auc(y, s) == pytest.approx(0.75)

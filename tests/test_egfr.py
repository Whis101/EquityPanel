"""Tests for equitypanel.reclassification.egfr: the CKD-EPI 2009 and 2021 equations.

Reference values are computed by hand from the published coefficients (Levey 2009,
Inker 2021); each test spells out the arithmetic so it can be checked on a calculator.
"""

import numpy as np
import pandas as pd
import pytest

from equitypanel.reclassification.egfr import egfr_ckd_epi_2009, egfr_ckd_epi_2021

# ---------------------------------------------------------------- reference values
# (creatinine, age, female, black, expected 2009, expected 2021)
# Rows 1-4 are the DOMAIN.md section 8 table. Values to 4 decimals.
REFERENCE_CASES = [
    # Black man, 55, Scr 1.4: G2 (no CKD) under 2009, G3a (CKD) under 2021
    (1.4, 55, False, True, 65.0911, 59.3568),
    # Non-Black man, 55, Scr 1.4: G3a both times
    (1.4, 55, False, False, 56.1614, 59.3568),
    # Black man, 55, Scr 3.5: above 20 under 2009, waitlist-eligible under 2021
    (3.5, 55, False, True, 21.4987, 19.7671),
    # Black woman, 60, Scr 1.2: G3a both times, lower number under 2021
    (1.2, 60, True, True, 56.8851, 51.8213),
    # Woman exactly at kappa (Scr 0.7): both min/max terms are 1
    (0.7, 40, True, False, 108.3769, 112.0543),
    # Woman below kappa: the alpha branch is used
    (0.5, 30, True, False, 129.8735, 129.3170),
    # Man below kappa (0.9), older
    (0.6, 70, False, False, 101.8685, 103.8478),
    # Man exactly at kappa
    (0.9, 50, False, False, 99.2388, 104.0490),
    # Black woman, 80, Scr 2.0
    (2.0, 80, True, True, 26.6544, 24.7896),
]


@pytest.mark.parametrize(("scr", "age", "female", "black", "exp_2009", "_"), REFERENCE_CASES)
def test_2009_reference_values(scr, age, female, black, exp_2009, _):
    got = egfr_ckd_epi_2009([scr], [age], [female], [black])
    assert got[0] == pytest.approx(exp_2009, abs=1e-3)


@pytest.mark.parametrize(("scr", "age", "female", "_", "__", "exp_2021"), REFERENCE_CASES)
def test_2021_reference_values(scr, age, female, _, __, exp_2021):
    got = egfr_ckd_epi_2021([scr], [age], [female])
    assert got[0] == pytest.approx(exp_2021, abs=1e-3)


def test_2009_black_man_arithmetic_spelled_out():
    # Scr 1.4 > kappa 0.9, so min term = 1 and max term = (1.4 / 0.9) ** -1.209
    expected = 141 * (1.4 / 0.9) ** -1.209 * 0.993**55 * 1.159
    got = egfr_ckd_epi_2009([1.4], [55], [False], [True])[0]
    assert got == pytest.approx(expected, rel=1e-12)
    assert round(got, 1) == 65.1


def test_2021_woman_below_kappa_arithmetic_spelled_out():
    # Scr 0.5 < kappa 0.7, so min term = (0.5 / 0.7) ** -0.241 and max term = 1
    expected = 142 * (0.5 / 0.7) ** -0.241 * 0.9938**30 * 1.012
    got = egfr_ckd_epi_2021([0.5], [30], [True])[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_2009_woman_below_kappa_arithmetic_spelled_out():
    expected = 141 * (0.5 / 0.7) ** -0.329 * 0.993**30 * 1.018
    got = egfr_ckd_epi_2009([0.5], [30], [True], [False])[0]
    assert got == pytest.approx(expected, rel=1e-12)


def test_2021_man_above_kappa_arithmetic_spelled_out():
    expected = 142 * (3.5 / 0.9) ** -1.200 * 0.9938**55
    got = egfr_ckd_epi_2021([3.5], [55], [False])[0]
    assert got == pytest.approx(expected, rel=1e-12)
    assert round(got, 1) == 19.8


# NKF's published test cases for labs implementing the 2021 equation (whole-number eGFR):
# kidney.org/sites/default/files/example_it_ticket-implement_2021_ckd-epi_equation_to_calculate_egfr_from_creatinine_1.pdf  # noqa: E501
# The female rows also catch a missing 1.012 factor (128 -> 127, 126 -> 125, 89 -> 88).
NKF_2021_CASES = [
    # (age, female, creatinine, expected eGFR rounded)
    (18, False, 0.90, 127),
    (18, False, 0.91, 125),
    (18, True, 0.70, 128),
    (18, True, 0.71, 126),
    (90, False, 0.50, 97),
    (90, False, 1.50, 44),
    (90, True, 0.50, 89),
    (90, True, 1.50, 33),
]


@pytest.mark.parametrize(("age", "female", "scr", "expected"), NKF_2021_CASES)
def test_2021_matches_nkf_published_test_cases(age, female, scr, expected):
    got = egfr_ckd_epi_2021([scr], [age], [female])[0]
    assert round(got) == expected


def test_2009_female_male_ratio_at_kappa_is_1_018():
    # At Scr = kappa both power terms are 1, so only the sex factor differs.
    female = egfr_ckd_epi_2009([0.7], [50], [True], [False])[0]
    male = egfr_ckd_epi_2009([0.9], [50], [False], [False])[0]
    assert female / male == pytest.approx(1.018, rel=1e-12)


# ---------------------------------------------------------------- properties


def test_at_kappa_only_constant_age_and_sex_terms_remain():
    # Scr = kappa: (Scr/kappa) = 1, so both power terms are 1.
    assert egfr_ckd_epi_2009([0.9], [18], [False], [False])[0] == pytest.approx(141 * 0.993**18)
    assert egfr_ckd_epi_2009([0.7], [18], [True], [False])[0] == pytest.approx(
        141 * 0.993**18 * 1.018
    )
    assert egfr_ckd_epi_2021([0.9], [18], [False])[0] == pytest.approx(142 * 0.9938**18)
    assert egfr_ckd_epi_2021([0.7], [18], [True])[0] == pytest.approx(142 * 0.9938**18 * 1.012)


@pytest.mark.parametrize("female", [False, True])
@pytest.mark.parametrize("scr", [0.4, 0.7, 0.9, 1.5, 4.0])
def test_2009_black_multiplier_is_exactly_1_159(scr, female):
    black = egfr_ckd_epi_2009([scr], [50], [female], [True])[0]
    other = egfr_ckd_epi_2009([scr], [50], [female], [False])[0]
    assert black / other == pytest.approx(1.159, rel=1e-12)


def test_falls_as_creatinine_rises():
    scr = np.linspace(0.3, 15, 200)
    for female in (False, True):
        flags = np.full(scr.shape, female)
        ages = np.full(scr.shape, 50)
        e09 = egfr_ckd_epi_2009(scr, ages, flags, np.zeros(scr.shape, dtype=bool))
        e21 = egfr_ckd_epi_2021(scr, ages, flags)
        assert np.all(np.diff(e09) < 0)
        assert np.all(np.diff(e21) < 0)


def test_falls_as_age_rises():
    ages = np.arange(18, 100)
    ones = np.ones(ages.shape)
    no = np.zeros(ages.shape, dtype=bool)
    assert np.all(np.diff(egfr_ckd_epi_2009(ones, ages, no, no)) < 0)
    assert np.all(np.diff(egfr_ckd_epi_2021(ones, ages, no)) < 0)


def test_continuous_at_kappa():
    # The two branches meet at Scr = kappa, so there is no jump either side of it.
    for female, kappa in ((True, 0.7), (False, 0.9)):
        scr = [kappa - 1e-9, kappa, kappa + 1e-9]
        e09 = egfr_ckd_epi_2009(scr, [50] * 3, [female] * 3, [False] * 3)
        e21 = egfr_ckd_epi_2021(scr, [50] * 3, [female] * 3)
        assert np.ptp(e09) < 1e-6
        assert np.ptp(e21) < 1e-6


def test_2021_ignores_race_by_construction():
    # Same creatinine, age and sex -> same 2021 eGFR (row 1 vs row 2 above).
    assert REFERENCE_CASES[0][5] == REFERENCE_CASES[1][5]


# ---------------------------------------------------------------- inputs and outputs


def test_vectorised_matches_one_at_a_time():
    scr, age, female, black, *_ = zip(*REFERENCE_CASES, strict=True)
    all_09 = egfr_ckd_epi_2009(scr, age, female, black)
    all_21 = egfr_ckd_epi_2021(scr, age, female)
    for i in range(len(scr)):
        assert all_09[i] == egfr_ckd_epi_2009([scr[i]], [age[i]], [female[i]], [black[i]])[0]
        assert all_21[i] == egfr_ckd_epi_2021([scr[i]], [age[i]], [female[i]])[0]


def test_returns_float_numpy_array_from_pandas_series():
    scr = pd.Series([1.0, 2.0], index=[10, 20])
    age = pd.Series([40.5, 61.2], index=[10, 20])
    female = pd.Series([True, False], index=[10, 20])
    got = egfr_ckd_epi_2021(scr, age, female)
    assert isinstance(got, np.ndarray)
    assert got.dtype == float
    assert got.shape == (2,)


def test_accepts_0_1_flags():
    a = egfr_ckd_epi_2009([1.1], [60], [1], [1])
    b = egfr_ckd_epi_2009([1.1], [60], [True], [True])
    np.testing.assert_array_equal(a, b)


def test_fractional_age_is_used_exactly():
    a = egfr_ckd_epi_2021([1.0], [50.5], [False])[0]
    b = egfr_ckd_epi_2021([1.0], [50.0], [False])[0]
    assert a / b == pytest.approx(0.9938**0.5)


@pytest.mark.parametrize(
    ("scr", "age", "female", "match"),
    [
        ([0.0], [50], [False], "positive"),
        ([-1.0], [50], [False], "positive"),
        ([np.nan], [50], [False], "missing"),
        ([1.0], [np.nan], [False], "missing"),
        ([1.0], [17.9], [False], "at least 18"),
        ([1.0], [50], [2], "True/False"),
        ([1.0, 2.0], [50], [False], "same shape"),
    ],
)
def test_invalid_inputs_raise(scr, age, female, match):
    with pytest.raises(ValueError, match=match):
        egfr_ckd_epi_2021(scr, age, female)
    with pytest.raises(ValueError, match=match):
        egfr_ckd_epi_2009(scr, age, female, [False] * len(scr))


def test_2009_invalid_black_flag_raises():
    with pytest.raises(ValueError, match="black"):
        egfr_ckd_epi_2009([1.0], [50], [False], ["yes"])

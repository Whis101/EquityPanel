"""Tests for equitypanel.reclassification.reclassify: stages, lines, per-group counts.

The six-patient lab frame below is chosen so every kind of change happens at least once.
Its eGFR values (from test_egfr.py's reference cases and the published coefficients):

  id  race   sex  age  Scr   2009     2021    stage        lines crossed (2009 -> 2021)
  p1  Black  M    55   3.5   21.50    19.77   G4 -> G4     newly <= 20 (waitlist)
  p2  Black  M    55   1.4   65.09    59.36   G2 -> G3a    newly < 60 (CKD)
  p3  White  M    55   1.35  58.69    62.00   G3a -> G2    newly >= 60 (CKD dropped)
  p4  Asian  M    55   3.3   19.92    21.21   G4 -> G4     newly > 20 (off waitlist)
  p5  White  M    50   0.9   99.24   104.05   G1 -> G1     none
  p6  Black  F    60   1.2   56.89    51.82   G3a -> G3a   none
"""

import json

import numpy as np
import pandas as pd
import pytest

from equitypanel.data.schema import SchemaError
from equitypanel.reclassification.reclassify import (
    DEFAULT_GROUP_COLS,
    LINES,
    STAGES,
    below_line,
    ckd_stage,
    filter_plausible,
    headline_numbers,
    reclassification_summary,
    reclassify,
    stage_crosstab,
)


def make_lab(**overrides) -> pd.DataFrame:
    lab = pd.DataFrame(
        {
            "patient_id": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "race": ["Black", "Black", "White", "Asian", "White", "Black"],
            "sex": ["Male", "Male", "Male", "Male", "Male", "Female"],
            "age_band": ["50-59", "50-59", "50-59", "50-59", "50-59", "60-69"],
            "age": [55.0, 55.0, 55.0, 55.0, 50.0, 60.0],
            "creatinine_mg_dl": [3.5, 1.4, 1.35, 3.3, 0.9, 1.2],
        }
    )
    for col, values in overrides.items():
        lab[col] = values
    return lab


@pytest.fixture(scope="module")
def patients():
    return reclassify(make_lab())


@pytest.fixture(scope="module")
def summary(patients):
    return reclassification_summary(patients)


def row(summary, col, group):
    hit = summary[(summary["group_col"] == col) & (summary["group"] == group)]
    assert len(hit) == 1
    return hit.iloc[0]


# ---------------------------------------------------------------- ckd_stage


@pytest.mark.parametrize(
    ("egfr", "stage"),
    [
        (130.0, "G1"),
        (90.0, "G1"),
        (89.999, "G2"),
        (60.0, "G2"),
        (59.999, "G3a"),
        (45.0, "G3a"),
        (44.9, "G3b"),
        (30.0, "G3b"),
        (29.9, "G4"),
        (15.0, "G4"),
        (14.99, "G5"),
        (2.0, "G5"),
    ],
)
def test_ckd_stage_boundaries(egfr, stage):
    assert ckd_stage(np.array([egfr]))[0] == stage


def test_ckd_stage_vectorised_and_accepts_series():
    got = ckd_stage(pd.Series([100.0, 50.0, 10.0]))
    assert list(got) == ["G1", "G3a", "G5"]


def test_ckd_stage_rejects_missing():
    with pytest.raises(ValueError, match="missing"):
        ckd_stage(np.array([50.0, np.nan]))


def test_stage_order_is_healthiest_first():
    assert STAGES == ("G1", "G2", "G3a", "G3b", "G4", "G5")


# ---------------------------------------------------------------- below_line


def test_lines_and_their_boundaries():
    assert set(LINES) == {"ckd_lt60", "referral_lt30", "waitlist_le20"}
    values = np.array([60.0, 59.99, 30.0, 29.99, 20.0, 20.01])
    assert list(below_line(values, "ckd_lt60")) == [False, True, True, True, True, True]
    assert list(below_line(values, "referral_lt30")) == [False, False, False, True, True, True]
    # The waitlist rule is "<= 20": exactly 20 counts.
    assert list(below_line(values, "waitlist_le20")) == [False, False, False, False, True, False]


# ---------------------------------------------------------------- filter_plausible


def test_filter_plausible_drops_outside_range_and_counts():
    lab = make_lab(creatinine_mg_dl=[0.19, 0.2, 1.0, 20.0, 20.01, 72.5])
    kept, dropped = filter_plausible(lab)
    assert dropped == 3
    assert list(kept["patient_id"]) == ["p2", "p3", "p4"]
    assert kept.index.equals(pd.RangeIndex(3))


def test_filter_plausible_does_not_modify_input():
    lab = make_lab()
    before = lab.copy()
    filter_plausible(lab)
    pd.testing.assert_frame_equal(lab, before)


# ---------------------------------------------------------------- reclassify


def test_reclassify_columns(patients):
    expected = [
        "patient_id",
        "race_term",
        "race",
        "sex",
        "age_band",
        "age",
        "creatinine_mg_dl",
        "egfr_2009",
        "egfr_2021",
        "stage_2009",
        "stage_2021",
        "direction",
    ]
    for line in LINES:
        expected += [f"{line}_2009", f"{line}_2021"]
    assert list(patients.columns) == expected
    assert len(patients) == 6


def test_reclassify_egfr_values(patients):
    np.testing.assert_allclose(
        patients["egfr_2009"], [21.4987, 65.0911, 58.6860, 19.9170, 99.2388, 56.8851], atol=1e-3
    )
    np.testing.assert_allclose(
        patients["egfr_2021"], [19.7671, 59.3568, 62.0050, 21.2130, 104.0490, 51.8213], atol=1e-3
    )


def test_reclassify_stages_and_direction(patients):
    assert list(patients["stage_2009"]) == ["G4", "G2", "G3a", "G4", "G1", "G3a"]
    assert list(patients["stage_2021"]) == ["G4", "G3a", "G2", "G4", "G1", "G3a"]
    assert list(patients["direction"]) == ["same", "lower", "higher", "same", "same", "same"]


def test_reclassify_race_term(patients):
    assert list(patients["race_term"]) == [
        "Black",
        "Black",
        "non-Black",
        "non-Black",
        "non-Black",
        "Black",
    ]


def test_reclassify_line_flags(patients):
    assert list(patients["waitlist_le20_2009"]) == [False, False, False, True, False, False]
    assert list(patients["waitlist_le20_2021"]) == [True, False, False, False, False, False]
    assert list(patients["ckd_lt60_2009"]) == [True, False, True, True, False, True]
    assert list(patients["ckd_lt60_2021"]) == [True, True, False, True, False, True]


def test_only_black_patients_get_the_2009_multiplier():
    # Same person labelled White instead of Black: 2009 falls by exactly 1.159; 2021 unchanged.
    black = reclassify(make_lab())
    white = reclassify(make_lab(race=["White"] * 6))
    ratio = black["egfr_2009"] / white["egfr_2009"]
    np.testing.assert_allclose(ratio[black["race"] == "Black"], 1.159)
    np.testing.assert_allclose(ratio[black["race"] != "Black"], 1.0)
    np.testing.assert_allclose(black["egfr_2021"], white["egfr_2021"])


def test_reclassify_does_not_modify_input():
    lab = make_lab()
    before = lab.copy()
    reclassify(lab)
    pd.testing.assert_frame_equal(lab, before)


def test_reclassify_keeps_input_order_with_any_index():
    lab = make_lab()
    lab.index = [50, 40, 30, 20, 10, 0]
    got = reclassify(lab)
    assert list(got["patient_id"]) == list(lab["patient_id"])
    assert got.index.equals(pd.RangeIndex(6))


def test_reclassify_rejects_unknown_sex():
    with pytest.raises(ValueError, match="Female or Male"):
        reclassify(make_lab(sex=["Male"] * 5 + ["Unknown"]))


def test_reclassify_validates_lab_frame():
    with pytest.raises(SchemaError):
        reclassify(make_lab().drop(columns="creatinine_mg_dl"))
    with pytest.raises(SchemaError):
        reclassify(make_lab(age=[55.0, 55.0, 55.0, 55.0, 50.0, 16.0]))


# ---------------------------------------------------------------- reclassification_summary


def test_summary_has_all_row_first_then_groups_in_order(summary):
    keys = list(zip(summary["group_col"], summary["group"], strict=True))
    assert keys == [
        ("all", "All"),
        ("race_term", "Black"),
        ("race_term", "non-Black"),
        ("race", "Asian"),
        ("race", "Black"),
        ("race", "White"),
        ("sex", "Female"),
        ("sex", "Male"),
        ("age_band", "50-59"),
        ("age_band", "60-69"),
    ]


def test_summary_default_group_cols():
    assert DEFAULT_GROUP_COLS == ("race_term", "race", "sex", "age_band")


def test_summary_all_row(summary):
    r = row(summary, "all", "All")
    assert r["n"] == 6
    assert (r["n_lower"], r["n_higher"], r["n_same"]) == (1, 1, 4)
    assert r["pct_lower"] == pytest.approx(1 / 6)
    assert r["pct_same"] == pytest.approx(4 / 6)


def test_summary_black_row(summary):
    r = row(summary, "race_term", "Black")
    assert r["n"] == 3
    assert (r["n_lower"], r["n_higher"], r["n_same"]) == (1, 0, 2)
    assert (r["ckd_lt60_2009"], r["ckd_lt60_2021"]) == (2, 3)
    assert (r["ckd_lt60_newly_below"], r["ckd_lt60_newly_above"]) == (1, 0)
    assert (r["referral_lt30_2009"], r["referral_lt30_2021"]) == (1, 1)
    assert (r["waitlist_le20_2009"], r["waitlist_le20_2021"]) == (0, 1)
    assert (r["waitlist_le20_newly_below"], r["waitlist_le20_newly_above"]) == (1, 0)


def test_summary_non_black_row(summary):
    r = row(summary, "race_term", "non-Black")
    assert r["n"] == 3
    assert (r["n_lower"], r["n_higher"], r["n_same"]) == (0, 1, 2)
    assert (r["ckd_lt60_2009"], r["ckd_lt60_2021"]) == (2, 1)
    assert (r["ckd_lt60_newly_below"], r["ckd_lt60_newly_above"]) == (0, 1)
    assert (r["waitlist_le20_2009"], r["waitlist_le20_2021"]) == (1, 0)
    assert (r["waitlist_le20_newly_below"], r["waitlist_le20_newly_above"]) == (0, 1)


def test_summary_newly_counts_reconcile(summary):
    # below under 2021 = below under 2009 + newly below - newly above, in every row
    for line in LINES:
        lhs = summary[f"{line}_2021"]
        rhs = (
            summary[f"{line}_2009"]
            + summary[f"{line}_newly_below"]
            - summary[f"{line}_newly_above"]
        )
        pd.testing.assert_series_equal(lhs, rhs, check_names=False)
    assert (summary["n_lower"] + summary["n_higher"] + summary["n_same"] == summary["n"]).all()


def test_summary_small_group_flag(summary):
    # Every group here has fewer than 30 patients.
    assert summary["small_group"].all()
    big = reclassify(
        pd.concat([make_lab(patient_id=[f"{k}-{i}" for i in range(6)]) for k in range(6)])
    )
    big_summary = reclassification_summary(big)
    assert not row(big_summary, "all", "All")["small_group"]  # n = 36
    assert row(big_summary, "sex", "Female")["small_group"]  # n = 6


def test_summary_custom_group_cols(patients):
    got = reclassification_summary(patients, group_cols=["sex"])
    assert list(got["group_col"]) == ["all", "sex", "sex"]


def test_summary_rejects_unknown_group_col_and_empty(patients):
    with pytest.raises(ValueError, match="not in df"):
        reclassification_summary(patients, group_cols=["income"])
    with pytest.raises(ValueError, match="empty"):
        reclassification_summary(patients.iloc[0:0])


# ---------------------------------------------------------------- stage_crosstab


def test_stage_crosstab(patients):
    table = stage_crosstab(patients)
    assert list(table.index) == list(STAGES)
    assert list(table.columns) == list(STAGES)
    assert table.to_numpy().sum() == 6
    assert table.loc["G4", "G4"] == 2
    assert table.loc["G2", "G3a"] == 1
    assert table.loc["G3a", "G2"] == 1
    assert table.loc["G1", "G1"] == 1
    assert table.loc["G3a", "G3a"] == 1
    assert table.loc["G5"].sum() == 0  # stages nobody is in still appear


# ---------------------------------------------------------------- headline_numbers


def test_headline_numbers_is_json_ready(summary):
    got = headline_numbers(summary)
    assert set(got) == {"Black", "non-Black"}
    assert got["Black"]["n"] == 3
    assert got["Black"]["waitlist_le20_newly_below"] == 1
    assert got["non-Black"]["waitlist_le20_newly_above"] == 1
    assert got["Black"]["small_group"] is True
    assert "group" not in got["Black"] and "group_col" not in got["Black"]
    json.dumps(got)  # plain Python types only


def test_headline_numbers_needs_both_groups(patients):
    only_black = reclassification_summary(patients[patients["race_term"] == "Black"])
    with pytest.raises(ValueError, match="both"):
        headline_numbers(only_black)

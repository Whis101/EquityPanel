"""Tests for equitypanel.data.uci, using small fake frames shaped like the raw UCI file."""

from pathlib import Path

import pandas as pd
import pytest

from equitypanel.data.schema import SchemaError, validate_cohort_frame
from equitypanel.data.uci import clean_uci, load_uci

REAL_UCI_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetic_data.csv"
RAW_REQUIRED_COLS = [
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "discharge_disposition_id",
    "readmitted",
]


def raw_row(**overrides) -> dict:
    """One raw UCI stay with sensible defaults. Override any column by keyword."""
    row = {
        "encounter_id": 1,
        "patient_nbr": 100,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "discharge_disposition_id": 1,
        "time_in_hospital": 3,
        "A1Cresult": "None",
        "readmitted": "NO",
    }
    row.update(overrides)
    return row


def make_raw(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------- output shape


def test_output_is_a_valid_cohort_frame():
    raw = make_raw(raw_row(), raw_row(encounter_id=2, patient_nbr=200, readmitted="<30"))
    cohort = clean_uci(raw)
    validate_cohort_frame(cohort)
    assert list(cohort.columns[:5]) == ["patient_id", "y_true", "race", "sex", "age_band"]


def test_feature_columns_pass_through_unchanged():
    cohort = clean_uci(make_raw(raw_row(time_in_hospital=7, A1Cresult=">8")))
    assert cohort.loc[0, "time_in_hospital"] == 7
    assert cohort.loc[0, "A1Cresult"] == ">8"
    assert cohort.loc[0, "discharge_disposition_id"] == 1


def test_raw_columns_that_were_mapped_are_removed():
    cohort = clean_uci(make_raw(raw_row()))
    for col in ["encounter_id", "patient_nbr", "gender", "age", "readmitted"]:
        assert col not in cohort.columns


def test_input_is_not_mutated():
    raw = make_raw(raw_row(), raw_row(encounter_id=2, discharge_disposition_id=11))
    before = raw.copy()
    clean_uci(raw)
    pd.testing.assert_frame_equal(raw, before)


# ---------------------------------------------------------------- value mapping


@pytest.mark.parametrize(("readmitted", "expected"), [("<30", 1), (">30", 0), ("NO", 0)])
def test_only_readmission_within_30_days_is_positive(readmitted, expected):
    cohort = clean_uci(make_raw(raw_row(readmitted=readmitted)))
    assert cohort.loc[0, "y_true"] == expected


@pytest.mark.parametrize(
    ("raw_race", "expected"),
    [
        ("Caucasian", "White"),
        ("AfricanAmerican", "Black"),
        ("Hispanic", "Hispanic"),
        ("Asian", "Asian"),
        ("Other", "Other"),
        ("?", "Unknown"),  # Bug A: kept as its own group, never dropped
    ],
)
def test_race_labels_are_normalised(raw_race, expected):
    cohort = clean_uci(make_raw(raw_row(race=raw_race)))
    assert cohort.loc[0, "race"] == expected


@pytest.mark.parametrize(
    ("gender", "expected"),
    [("Female", "Female"), ("Male", "Male"), ("Unknown/Invalid", "Unknown")],
)
def test_gender_maps_to_sex(gender, expected):
    cohort = clean_uci(make_raw(raw_row(gender=gender)))
    assert cohort.loc[0, "sex"] == expected


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        ("[0-10)", "<50"),
        ("[10-20)", "<50"),
        ("[20-30)", "<50"),
        ("[30-40)", "<50"),
        ("[40-50)", "<50"),
        ("[50-60)", "50-59"),
        ("[60-70)", "60-69"),
        ("[70-80)", "70-79"),
        ("[80-90)", "80+"),
        ("[90-100)", "80+"),
    ],
)
def test_age_brackets_collapse_to_bands(age, expected):
    cohort = clean_uci(make_raw(raw_row(age=age)))
    assert cohort.loc[0, "age_band"] == expected


# ---------------------------------------------------------------- row selection


def test_keeps_only_each_patients_first_stay():
    # Bug C: one row per patient. Rows are deliberately out of order.
    raw = make_raw(
        raw_row(encounter_id=5, patient_nbr=100, readmitted="<30", time_in_hospital=9),
        raw_row(encounter_id=2, patient_nbr=100, readmitted="NO", time_in_hospital=4),
        raw_row(encounter_id=3, patient_nbr=200),
    )
    cohort = clean_uci(raw)
    assert sorted(cohort["patient_id"]) == [100, 200]
    first = cohort[cohort["patient_id"] == 100].iloc[0]
    assert first["time_in_hospital"] == 4
    assert first["y_true"] == 0


@pytest.mark.parametrize("disposition", [11, 13, 14, 19, 20, 21])
def test_drops_stays_ending_in_death_or_hospice(disposition):
    raw = make_raw(
        raw_row(encounter_id=1, patient_nbr=100, discharge_disposition_id=disposition),
        raw_row(encounter_id=2, patient_nbr=200, discharge_disposition_id=1),
    )
    assert list(clean_uci(raw)["patient_id"]) == [200]


def test_first_eligible_stay_is_used_when_an_earlier_one_is_dropped():
    raw = make_raw(
        raw_row(encounter_id=1, patient_nbr=100, discharge_disposition_id=13),
        raw_row(encounter_id=2, patient_nbr=100, discharge_disposition_id=1, time_in_hospital=6),
    )
    cohort = clean_uci(raw)
    assert len(cohort) == 1
    assert cohort.loc[0, "time_in_hospital"] == 6


def test_no_eligible_stays_fails_the_schema():
    raw = make_raw(raw_row(discharge_disposition_id=11))
    with pytest.raises(SchemaError, match="empty"):
        clean_uci(raw)


# ---------------------------------------------------------------- bad raw input


@pytest.mark.parametrize(
    ("col", "bad"),
    [("race", "Martian"), ("gender", "X"), ("age", "[100-110)"), ("readmitted", "maybe")],
)
def test_unexpected_raw_value_is_rejected(col, bad):
    # Without this, the value would silently become NaN and the schema error
    # would blame "missing values" instead of the real problem.
    with pytest.raises(ValueError, match=col):
        clean_uci(make_raw(raw_row(**{col: bad})))


@pytest.mark.parametrize("col", RAW_REQUIRED_COLS)
def test_missing_raw_column_is_rejected(col):
    with pytest.raises(ValueError, match=col):
        clean_uci(make_raw(raw_row()).drop(columns=col))


# ---------------------------------------------------------------- reading the CSV


def test_load_uci_reads_a_csv_and_keeps_none_as_text(tmp_path):
    # Bug D: read_csv turns the text "None" into NaN by default, but in UCI
    # A1Cresult == "None" means "test not done", which is real information.
    path = tmp_path / "diabetic_data.csv"
    make_raw(raw_row(race="?"), raw_row(encounter_id=2, patient_nbr=200)).to_csv(path, index=False)
    cohort = load_uci(path)
    assert (cohort["A1Cresult"] == "None").all()
    assert cohort.loc[0, "race"] == "Unknown"


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_UCI_PATH.exists(), reason="data/raw/diabetic_data.csv not downloaded")
def test_real_uci_file_loads():
    cohort = load_uci(REAL_UCI_PATH)
    assert len(cohort) == 69_990  # 71,518 patients minus those with no eligible stay
    assert 0.08 < cohort["y_true"].mean() < 0.10  # about 9% readmitted within 30 days

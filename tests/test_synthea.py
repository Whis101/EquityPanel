"""Tests for equitypanel.data.synthea, using small fake frames shaped like Synthea's CSVs."""

from pathlib import Path

import pandas as pd
import pytest

from equitypanel.data.schema import LAB_REQUIRED_COLS, SchemaError, validate_lab_frame
from equitypanel.data.synthea import clean_synthea, load_synthea

REAL_SYNTHEA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "synthea"
PATIENT_REQUIRED_COLS = ["Id", "BIRTHDATE", "RACE", "GENDER"]
OBSERVATION_REQUIRED_COLS = ["DATE", "PATIENT", "CODE", "VALUE", "UNITS"]


def patient(**overrides) -> dict:
    """One Synthea patient with sensible defaults. Override any column by keyword."""
    row = {
        "Id": "p1",
        "BIRTHDATE": "1960-01-01",
        "DEATHDATE": "",
        "RACE": "white",
        "ETHNICITY": "nonhispanic",
        "GENDER": "F",
    }
    row.update(overrides)
    return row


def obs(**overrides) -> dict:
    """One Synthea observation: by default a serum creatinine test at age 60."""
    row = {
        "DATE": "2020-01-01",
        "PATIENT": "p1",
        "ENCOUNTER": "e1",
        "CODE": "2160-0",
        "DESCRIPTION": "Creatinine [Mass/volume] in Serum or Plasma",
        "VALUE": "1.2",
        "UNITS": "mg/dL",
        "TYPE": "numeric",
    }
    row.update(overrides)
    return row


def frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows))


# ---------------------------------------------------------------- output shape


def test_output_is_a_valid_lab_frame():
    lab = clean_synthea(frame(patient()), frame(obs()))
    validate_lab_frame(lab)
    assert list(lab.columns) == list(LAB_REQUIRED_COLS)


def test_creatinine_text_becomes_a_number():
    lab = clean_synthea(frame(patient()), frame(obs(VALUE="1.75")))
    assert lab.loc[0, "creatinine_mg_dl"] == pytest.approx(1.75)


def test_inputs_are_not_mutated():
    patients = frame(patient())
    observations = frame(obs(), obs(DATE="1970-01-01"))
    before_p, before_o = patients.copy(), observations.copy()
    clean_synthea(patients, observations)
    pd.testing.assert_frame_equal(patients, before_p)
    pd.testing.assert_frame_equal(observations, before_o)


# ---------------------------------------------------------------- value mapping


@pytest.mark.parametrize(
    ("raw_race", "expected"),
    [
        ("white", "White"),
        ("black", "Black"),
        ("asian", "Asian"),
        ("native", "Native"),
        ("hawaiian", "Pacific Islander"),
        ("other", "Other"),
    ],
)
def test_race_labels_are_normalised(raw_race, expected):
    lab = clean_synthea(frame(patient(RACE=raw_race)), frame(obs()))
    assert lab.loc[0, "race"] == expected


def test_ethnicity_is_ignored():
    # Synthea stores Hispanic as an ethnicity, separate from race.
    lab = clean_synthea(frame(patient(ETHNICITY="hispanic")), frame(obs()))
    assert lab.loc[0, "race"] == "White"
    assert "ETHNICITY" not in lab.columns


@pytest.mark.parametrize(("gender", "expected"), [("F", "Female"), ("M", "Male")])
def test_gender_maps_to_sex(gender, expected):
    lab = clean_synthea(frame(patient(GENDER=gender)), frame(obs()))
    assert lab.loc[0, "sex"] == expected


def test_age_is_exact_years_at_the_test_date():
    # eGFR formulas need the real age, not a 10-year band.
    lab = clean_synthea(frame(patient(BIRTHDATE="1960-01-01")), frame(obs(DATE="2020-07-01")))
    assert lab.loc[0, "age"] == pytest.approx(60.5, abs=0.01)


@pytest.mark.parametrize(
    ("birthdate", "expected"),
    [
        ("2001-06-01", "<50"),  # 18.6
        ("1971-01-01", "<50"),  # 49.0
        ("1969-06-01", "50-59"),  # 50.6
        ("1955-01-01", "60-69"),  # 65.0
        ("1945-01-01", "70-79"),  # 75.0
        ("1939-06-01", "80+"),  # 80.6
        ("1920-01-01", "80+"),  # 100.0
    ],
)
def test_age_band_comes_from_the_exact_age(birthdate, expected):
    lab = clean_synthea(frame(patient(BIRTHDATE=birthdate)), frame(obs(DATE="2020-01-01")))
    assert lab.loc[0, "age_band"] == expected


def test_dates_with_a_time_and_time_zone_are_read():
    # Older Synthea exports write "2020-01-01"; newer ones add a time and "Z" (UTC).
    observations = frame(obs(DATE="2020-07-01T08:35:36Z"))
    lab = clean_synthea(frame(patient(BIRTHDATE="1960-01-01")), observations)
    assert lab.loc[0, "age"] == pytest.approx(60.5, abs=0.01)


# ---------------------------------------------------------------- which tests count


def test_blood_creatinine_code_is_also_used():
    lab = clean_synthea(frame(patient()), frame(obs(CODE="38483-4", VALUE="0.9")))
    assert lab.loc[0, "creatinine_mg_dl"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    "code",
    [
        "14959-1",  # microalbumin/creatinine ratio in urine (mg/g)
        "33914-3",  # eGFR itself
        "4548-4",  # hemoglobin A1c
    ],
)
def test_other_codes_are_ignored(code):
    observations = frame(obs(PATIENT="p1"), obs(PATIENT="p2", CODE=code, UNITS="other"))
    lab = clean_synthea(frame(patient(Id="p1"), patient(Id="p2")), observations)
    assert list(lab["patient_id"]) == ["p1"]


def test_keeps_each_patients_latest_adult_test():
    # Rows are deliberately out of date order.
    observations = frame(
        obs(DATE="2019-01-01", VALUE="1.1"),
        obs(DATE="2021-01-01", VALUE="2.4"),
        obs(DATE="2020-01-01", VALUE="1.7"),
    )
    lab = clean_synthea(frame(patient()), observations)
    assert len(lab) == 1
    assert lab.loc[0, "creatinine_mg_dl"] == pytest.approx(2.4)


def test_childhood_tests_are_ignored():
    # The CKD-EPI equations are for adults only.
    patients = frame(patient(Id="adult"), patient(Id="child", BIRTHDATE="2010-01-01"))
    observations = frame(obs(PATIENT="adult"), obs(PATIENT="child"))
    lab = clean_synthea(patients, observations)
    assert list(lab["patient_id"]) == ["adult"]


def test_patients_without_creatinine_are_dropped():
    patients = frame(patient(Id="p1"), patient(Id="p2"))
    lab = clean_synthea(patients, frame(obs(PATIENT="p1")))
    assert list(lab["patient_id"]) == ["p1"]


def test_no_adult_creatinine_at_all_fails_the_schema():
    patients = frame(patient(BIRTHDATE="2010-01-01"))
    with pytest.raises(SchemaError, match="empty"):
        clean_synthea(patients, frame(obs()))


# ---------------------------------------------------------------- bad raw input


@pytest.mark.parametrize(("col", "bad"), [("RACE", "martian"), ("GENDER", "X")])
def test_unexpected_raw_value_is_rejected(col, bad):
    with pytest.raises(ValueError, match=col):
        clean_synthea(frame(patient(**{col: bad})), frame(obs()))


def test_creatinine_in_other_units_is_rejected():
    # umol/L values are about 88 times larger than mg/dL; mixing them would be silent.
    with pytest.raises(ValueError, match="UNITS"):
        clean_synthea(frame(patient()), frame(obs(UNITS="umol/L", VALUE="106")))


def test_non_numeric_creatinine_is_rejected():
    with pytest.raises(ValueError, match="VALUE"):
        clean_synthea(frame(patient()), frame(obs(VALUE="high")))


@pytest.mark.parametrize("col", PATIENT_REQUIRED_COLS)
def test_missing_patient_column_is_rejected(col):
    with pytest.raises(ValueError, match=col):
        clean_synthea(frame(patient()).drop(columns=col), frame(obs()))


@pytest.mark.parametrize("col", OBSERVATION_REQUIRED_COLS)
def test_missing_observation_column_is_rejected(col):
    with pytest.raises(ValueError, match=col):
        clean_synthea(frame(patient()), frame(obs()).drop(columns=col))


# ---------------------------------------------------------------- reading the CSVs


def test_load_synthea_reads_a_folder_of_csvs(tmp_path):
    frame(patient(Id="p1"), patient(Id="p2", RACE="black")).to_csv(
        tmp_path / "patients.csv", index=False
    )
    frame(obs(PATIENT="p1"), obs(PATIENT="p2", VALUE="0.8")).to_csv(
        tmp_path / "observations.csv", index=False
    )
    lab = load_synthea(tmp_path)
    assert sorted(lab["patient_id"]) == ["p1", "p2"]
    assert lab.set_index("patient_id").loc["p2", "race"] == "Black"


@pytest.mark.real_data
@pytest.mark.skipif(
    not (REAL_SYNTHEA_DIR / "observations.csv").exists(),
    reason="data/raw/synthea/ (10k Synthea CSV export) not downloaded",
)
def test_real_synthea_export_loads():
    lab = load_synthea(REAL_SYNTHEA_DIR)
    assert len(lab) == 5_507  # adults with at least one creatinine test
    assert (lab["race"] == "Black").sum() == 542

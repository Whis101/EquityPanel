"""Loader for Synthea synthetic patients, used only for serum creatinine.

Synthea writes one CSV per record type. clean_synthea takes two of them,
patients.csv and observations.csv, and turns them into a lab frame: one row per
adult patient with their latest creatinine test, plus race, sex, exact age at
that test and age band. The values are simulated, so any result built on them
must be labelled as synthetic.
"""

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from equitypanel.data.schema import ADULT_AGE, AGE_BANDS, validate_lab_frame

PATIENT_REQUIRED_COLS = ("Id", "BIRTHDATE", "RACE", "GENDER")
OBSERVATION_REQUIRED_COLS = ("DATE", "PATIENT", "CODE", "VALUE", "UNITS")

# LOINC codes Synthea uses for creatinine: 2160-0 is serum or plasma,
# 38483-4 is blood. Other codes that mention creatinine (urine ratios, eGFR
# itself) are deliberately left out.
CREATININE_CODES = ("2160-0", "38483-4")
CREATININE_UNITS = "mg/dL"

RACE_LABELS = {
    "white": "White",
    "black": "Black",
    "asian": "Asian",
    "native": "Native",
    "hawaiian": "Pacific Islander",
    "other": "Other",
}

GENDER_TO_SEX = {"F": "Female", "M": "Male"}

DAYS_PER_YEAR = 365.25

# Edges of AGE_BANDS in years; the first band starts at the adult age.
BAND_EDGES = (ADULT_AGE, 50, 60, 70, 80, np.inf)


def _require_columns(df: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing column(s): {missing}")


def _map_strict(values: pd.Series, mapping: dict, col: str) -> pd.Series:
    """Map values through mapping, raising ValueError on any value it doesn't cover."""
    unexpected = values[~values.isin(list(mapping))].unique().tolist()
    if unexpected:
        raise ValueError(f"column {col!r} has unexpected raw values {unexpected}")
    return values.map(mapping)


def _to_naive_datetime(values: pd.Series) -> pd.Series:
    # Older Synthea exports write "2019-08-01", newer ones "2019-08-01T10:41:04Z".
    # Parse both as UTC, then drop the time zone so dates compare with each other.
    return pd.to_datetime(values, format="ISO8601", utc=True).dt.tz_localize(None)


def _age_band(age: pd.Series) -> pd.Series:
    bands = pd.cut(age, bins=list(BAND_EDGES), labels=list(AGE_BANDS), right=False)
    return bands.astype(str)


def clean_synthea(patients: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Turn Synthea patients and observations into a validated lab frame.

    Keeps creatinine tests taken at age 18 or older, then each patient's
    latest one. Patients with no such test are dropped. Does not modify its
    inputs.
    """
    _require_columns(patients, PATIENT_REQUIRED_COLS, "patients")
    _require_columns(observations, OBSERVATION_REQUIRED_COLS, "observations")

    tests = observations[observations["CODE"].isin(CREATININE_CODES)]
    bad_units = tests.loc[tests["UNITS"] != CREATININE_UNITS, "UNITS"].unique().tolist()
    if bad_units:
        raise ValueError(f"creatinine UNITS must be {CREATININE_UNITS!r}, found {bad_units}")
    values = pd.to_numeric(tests["VALUE"], errors="coerce")
    if values.isna().any():
        bad = tests.loc[values.isna(), "VALUE"].unique().tolist()
        raise ValueError(f"creatinine VALUE must be numeric, found {bad}")

    measured = pd.DataFrame(
        {
            "patient_id": tests["PATIENT"],
            "test_date": _to_naive_datetime(tests["DATE"]),
            "creatinine_mg_dl": values.astype(float),
        }
    )
    people = patients[list(PATIENT_REQUIRED_COLS)].rename(columns={"Id": "patient_id"})
    merged = measured.merge(people, on="patient_id", how="inner", validate="many_to_one")
    birth = _to_naive_datetime(merged["BIRTHDATE"])
    merged["age"] = (merged["test_date"] - birth).dt.days / DAYS_PER_YEAR

    adult = merged[merged["age"] >= ADULT_AGE]
    latest = adult.sort_values("test_date", kind="stable").drop_duplicates(
        "patient_id", keep="last"
    )

    lab = pd.DataFrame(
        {
            "patient_id": latest["patient_id"],
            "race": _map_strict(latest["RACE"], RACE_LABELS, "RACE"),
            "sex": _map_strict(latest["GENDER"], GENDER_TO_SEX, "GENDER"),
            "age_band": _age_band(latest["age"]),
            "age": latest["age"],
            "creatinine_mg_dl": latest["creatinine_mg_dl"],
        }
    ).reset_index(drop=True)
    return validate_lab_frame(lab)


def load_synthea(folder: str | PathLike) -> pd.DataFrame:
    """Read patients.csv and observations.csv from a Synthea CSV export folder."""
    folder = Path(folder)
    patients = pd.read_csv(folder / "patients.csv", usecols=list(PATIENT_REQUIRED_COLS), dtype=str)
    observations = pd.read_csv(
        folder / "observations.csv", usecols=list(OBSERVATION_REQUIRED_COLS), dtype=str
    )
    return clean_synthea(patients, observations)

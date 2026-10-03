"""Loader for the UCI "Diabetes 130-US Hospitals (1999-2008)" dataset.

The raw file has one row per hospital stay (encounter). clean_uci turns it into a
cohort frame: one row per patient, with the label y_true (readmitted within 30
days), the group columns race, sex and age_band, and every other raw column
passed through as a feature.
"""

from os import PathLike

import pandas as pd

from equitypanel.data.schema import validate_cohort_frame

RAW_REQUIRED_COLS = (
    "encounter_id",
    "patient_nbr",
    "race",
    "gender",
    "age",
    "discharge_disposition_id",
    "readmitted",
)

# Discharge codes for stays that ended in death or hospice. These patients
# can't be readmitted, so keeping them would add fake "not readmitted" rows.
DEATH_OR_HOSPICE_DISPOSITIONS = (11, 13, 14, 19, 20, 21)

READMITTED_TO_Y_TRUE = {"<30": 1, ">30": 0, "NO": 0}

RACE_LABELS = {
    "Caucasian": "White",
    "AfricanAmerican": "Black",
    "Hispanic": "Hispanic",
    "Asian": "Asian",
    "Other": "Other",
    "?": "Unknown",
}

GENDER_TO_SEX = {"Female": "Female", "Male": "Male", "Unknown/Invalid": "Unknown"}

AGE_TO_BAND = {
    "[0-10)": "<50",
    "[10-20)": "<50",
    "[20-30)": "<50",
    "[30-40)": "<50",
    "[40-50)": "<50",
    "[50-60)": "50-59",
    "[60-70)": "60-69",
    "[70-80)": "70-79",
    "[80-90)": "80+",
    "[90-100)": "80+",
}

# Raw columns that are replaced by a mapped cohort column.
MAPPED_RAW_COLS = ("encounter_id", "patient_nbr", "gender", "age", "readmitted")


def _map_strict(values: pd.Series, mapping: dict, col: str) -> pd.Series:
    """Map values through mapping, raising ValueError on any value it doesn't cover."""
    unexpected = values[~values.isin(list(mapping))].unique().tolist()
    if unexpected:
        raise ValueError(f"column {col!r} has unexpected raw values {unexpected}")
    return values.map(mapping)


def clean_uci(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw UCI encounter table into a validated cohort frame.

    Drops stays that ended in death or hospice, then keeps each patient's
    first remaining stay (lowest encounter_id). Does not modify raw.
    """
    missing = [col for col in RAW_REQUIRED_COLS if col not in raw.columns]
    if missing:
        raise ValueError(f"raw UCI frame is missing column(s): {missing}")

    eligible = raw[~raw["discharge_disposition_id"].isin(DEATH_OR_HOSPICE_DISPOSITIONS)]
    first_stays = eligible.sort_values("encounter_id").drop_duplicates("patient_nbr")

    cohort = pd.DataFrame(
        {
            "patient_id": first_stays["patient_nbr"],
            "y_true": _map_strict(first_stays["readmitted"], READMITTED_TO_Y_TRUE, "readmitted"),
            "race": _map_strict(first_stays["race"], RACE_LABELS, "race"),
            "sex": _map_strict(first_stays["gender"], GENDER_TO_SEX, "gender"),
            "age_band": _map_strict(first_stays["age"], AGE_TO_BAND, "age"),
        }
    )
    features = first_stays.drop(columns=[*MAPPED_RAW_COLS, "race"])
    cohort = pd.concat([cohort, features], axis=1).reset_index(drop=True)
    return validate_cohort_frame(cohort)


def load_uci(path: str | PathLike) -> pd.DataFrame:
    """Read the raw UCI CSV at path and return a validated cohort frame."""
    # keep_default_na=False: otherwise read_csv turns the text "None" (e.g.
    # A1Cresult "test not done") into NaN. The file marks missing data with "?".
    raw = pd.read_csv(path, keep_default_na=False, low_memory=False)
    return clean_uci(raw)

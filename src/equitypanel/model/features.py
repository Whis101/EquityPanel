"""Turn a UCI cohort frame into model inputs.

Race and sex are never inputs: the audit checks whether the model is fair without
seeing them. Insurance (payer_code) is left out as a proxy for income and race.
"""

import re

import pandas as pd

NUMERIC_FEATURES = (
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
)
CATEGORICAL_FEATURES = (
    "age_band",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "metformin",
    "change",
    "diabetesMed",
)
DIAG_COLS = ("diag_1", "diag_2", "diag_3")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + DIAG_COLS

# ICD-9 groups from Strack et al. 2014, as [low, high) ranges. Diabetes (250.xx)
# is checked first; anything not listed, including V and E codes, is "other".
ICD9_RANGES = (
    ("circulatory", ((390, 460), (785, 786))),
    ("respiratory", ((460, 520), (786, 787))),
    ("digestive", ((520, 580), (787, 788))),
    ("injury", ((800, 1000),)),
    ("musculoskeletal", ((710, 740),)),
    ("genitourinary", ((580, 630), (788, 789))),
    ("neoplasms", ((140, 240),)),
)
_NUMERIC_CODE = re.compile(r"\d+(\.\d+)?")
_V_OR_E_CODE = re.compile(r"[VE]\d+(\.\d+)?")


def icd9_group(code: str) -> str:
    """Map one ICD-9 diagnosis code to its group. "?" (not recorded) is "missing".

    Raises ValueError for anything that isn't an ICD-9 code, so a new format
    fails loudly instead of being lumped into "other".
    """
    if not isinstance(code, str):
        raise ValueError(f"not an ICD-9 code: {code!r}")
    if code == "?":
        return "missing"
    if _V_OR_E_CODE.fullmatch(code):
        return "other"
    if not _NUMERIC_CODE.fullmatch(code):
        raise ValueError(f"not an ICD-9 code: {code!r}")
    x = float(code)
    if 250 <= x < 251:
        return "diabetes"
    for group, ranges in ICD9_RANGES:
        if any(low <= x < high for low, high in ranges):
            return group
    return "other"


def build_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """Return the model inputs: FEATURE_COLUMNS, same rows and index as cohort.

    Counts become float, categories become str (the *_id columns are labels, not
    quantities), and diagnoses become their ICD-9 group. Does not modify cohort.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in cohort.columns]
    if missing:
        raise ValueError(f"cohort is missing feature column(s): {missing}")

    X = pd.DataFrame(index=cohort.index)
    for col in NUMERIC_FEATURES:
        X[col] = cohort[col].astype("float64")
    for col in CATEGORICAL_FEATURES:
        X[col] = cohort[col].astype(str)
    for col in DIAG_COLS:
        codes = cohort[col].astype(str)
        groups = {code: icd9_group(code) for code in codes.unique()}  # ~900 codes, not 70k calls
        X[col] = codes.map(groups)
    return X

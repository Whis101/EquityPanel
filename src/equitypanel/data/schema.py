"""Schema contracts for the frames that flow through EquityPanel.

- Cohort frame: what the UCI loader returns. One row per patient, with the
  label, the demographic group columns and the model features.
- Audit frame: what metrics take. The label, the model's predicted
  probability and the group columns to compare.
- Lab frame: what the Synthea loader returns. One row per adult patient, with
  the group columns, exact age and one creatinine value. It has no label: it
  feeds the eGFR formulas, not a model audit.

Each validator checks its frame, raises SchemaError on the first problem it
finds and otherwise returns the same frame unchanged. Validators never modify
their input.
"""

from collections.abc import Sequence

import pandas as pd

AGE_BANDS = ("<50", "50-59", "60-69", "70-79", "80+")
SEX_VALUES = ("Female", "Male", "Unknown")

COHORT_GROUP_COLS = ("race", "sex", "age_band")
COHORT_REQUIRED_COLS = ("patient_id", "y_true", *COHORT_GROUP_COLS)

ADULT_AGE = 18
LAB_REQUIRED_COLS = ("patient_id", *COHORT_GROUP_COLS, "age", "creatinine_mg_dl")


class SchemaError(ValueError):
    """A data frame broke its schema contract."""


# ---------------------------------------------------------------- checks


def _check_not_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise SchemaError("frame is empty")


def _check_columns_present(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise SchemaError(f"missing required column(s): {missing}")


def _check_binary(df: pd.DataFrame, col: str) -> None:
    values = df[col]
    is_binary = pd.api.types.is_numeric_dtype(values) and values.isin([0, 1]).all()
    if not is_binary:
        raise SchemaError(f"{col} must contain only 0 and 1, with no missing values")


def _check_no_missing(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for col in cols:
        if df[col].isna().any():
            raise SchemaError(f"group column {col!r} has missing values")


def _check_allowed_values(df: pd.DataFrame, col: str, allowed: Sequence[str]) -> None:
    unexpected = df.loc[~df[col].isin(allowed), col].unique().tolist()
    if unexpected:
        raise SchemaError(f"{col} has unexpected values {unexpected}; allowed: {list(allowed)}")


# ---------------------------------------------------------------- validators


def validate_cohort_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check that df is a valid cohort frame and return it unchanged.

    Raises SchemaError if the frame is empty, a required column is missing,
    patient_id repeats, y_true is not 0/1, a group column has missing values,
    or age_band or sex holds a value outside the agreed set.
    """
    _check_not_empty(df)
    _check_columns_present(df, COHORT_REQUIRED_COLS)
    if df["patient_id"].duplicated().any():
        raise SchemaError("patient_id must be unique: one row per patient")
    _check_binary(df, "y_true")
    _check_no_missing(df, COHORT_GROUP_COLS)
    _check_allowed_values(df, "age_band", AGE_BANDS)
    _check_allowed_values(df, "sex", SEX_VALUES)
    return df


def validate_audit_frame(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Check that df is a valid audit frame for group_cols and return it unchanged.

    Raises SchemaError if group_cols is empty, the frame is empty, a required
    column is missing, y_true is not 0/1 or has only one class, y_score is not
    a probability in [0, 1], or a group column has missing values.
    """
    if len(group_cols) == 0:
        raise SchemaError("group_cols must name at least one column")
    _check_not_empty(df)
    _check_columns_present(df, ["y_true", "y_score", *group_cols])
    _check_binary(df, "y_true")
    if df["y_true"].nunique() < 2:
        raise SchemaError("y_true has only one class; need both 0 and 1")
    scores = df["y_score"]
    is_probability = (
        pd.api.types.is_numeric_dtype(scores)
        and scores.notna().all()
        and scores.between(0, 1).all()
    )
    if not is_probability:
        raise SchemaError("y_score must be numeric probabilities in [0, 1], with no missing values")
    _check_no_missing(df, group_cols)
    return df


def validate_lab_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Check that df is a valid lab frame and return it unchanged.

    Raises SchemaError if the frame is empty, a required column is missing,
    patient_id repeats, age is not a number >= 18, creatinine_mg_dl is not a
    number > 0, a group column has missing values, or age_band or sex holds a
    value outside the agreed set.
    """
    _check_not_empty(df)
    _check_columns_present(df, LAB_REQUIRED_COLS)
    if df["patient_id"].duplicated().any():
        raise SchemaError("patient_id must be unique: one row per patient")
    ages = df["age"]
    if not (pd.api.types.is_numeric_dtype(ages) and (ages >= ADULT_AGE).all()):
        raise SchemaError(f"age must be numeric years >= {ADULT_AGE}, with no missing values")
    creatinine = df["creatinine_mg_dl"]
    if not (pd.api.types.is_numeric_dtype(creatinine) and (creatinine > 0).all()):
        raise SchemaError("creatinine_mg_dl must be numeric and > 0, with no missing values")
    _check_no_missing(df, COHORT_GROUP_COLS)
    _check_allowed_values(df, "age_band", AGE_BANDS)
    _check_allowed_values(df, "sex", SEX_VALUES)
    return df

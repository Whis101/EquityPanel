"""Tests for equitypanel.data.schema: the cohort-frame and audit-frame contracts."""

import numpy as np
import pandas as pd
import pytest

from equitypanel.data.schema import (
    AGE_BANDS,
    SEX_VALUES,
    SchemaError,
    validate_audit_frame,
    validate_cohort_frame,
)

GROUP_COLS = ["race", "sex", "age_band"]


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def cohort() -> pd.DataFrame:
    """A small, valid cohort frame: one row per patient, plus one feature column."""
    return pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104],
            "y_true": [1, 0, 0, 1],
            "race": ["Black", "White", "Unknown", "Hispanic"],
            "sex": ["Female", "Male", "Unknown", "Female"],
            "age_band": ["<50", "50-59", "70-79", "80+"],
            "num_medications": [14, 6, 9, 21],
        }
    )


@pytest.fixture
def audit() -> pd.DataFrame:
    """A small, valid audit frame. Scores include the edge values 0.0 and 1.0."""
    return pd.DataFrame(
        {
            "y_true": [1, 0, 0, 1],
            "y_score": [0.72, 0.15, 0.0, 1.0],
            "race": ["Black", "White", "Unknown", "Hispanic"],
            "sex": ["Female", "Male", "Unknown", "Female"],
            "age_band": ["<50", "50-59", "70-79", "80+"],
        }
    )


def blank_first_row(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a copy of frame with col set to missing (NaN) in the first row."""
    out = frame.copy()
    out[col] = out[col].where(out.index != 0)
    return out


# ---------------------------------------------------------------- constants


def test_schema_error_is_a_value_error():
    assert issubclass(SchemaError, ValueError)


def test_age_bands_are_the_agreed_five():
    assert AGE_BANDS == ("<50", "50-59", "60-69", "70-79", "80+")


def test_sex_values_are_the_agreed_three():
    assert set(SEX_VALUES) == {"Female", "Male", "Unknown"}


# ---------------------------------------------------------------- cohort frame


def test_valid_cohort_is_returned_as_the_same_object(cohort):
    assert validate_cohort_frame(cohort) is cohort


def test_cohort_is_not_mutated(cohort):
    before = cohort.copy()
    validate_cohort_frame(cohort)
    pd.testing.assert_frame_equal(cohort, before)


def test_empty_cohort_is_rejected(cohort):
    with pytest.raises(SchemaError, match="empty"):
        validate_cohort_frame(cohort.iloc[0:0])


@pytest.mark.parametrize("col", ["patient_id", "y_true", "race", "sex", "age_band"])
def test_cohort_missing_required_column_is_rejected(cohort, col):
    with pytest.raises(SchemaError, match=col):
        validate_cohort_frame(cohort.drop(columns=col))


def test_duplicate_patient_id_is_rejected(cohort):
    # Bug C: a repeated patient makes bootstrap CIs falsely narrow.
    cohort["patient_id"] = [101, 101, 103, 104]
    with pytest.raises(SchemaError, match="patient_id"):
        validate_cohort_frame(cohort)


@pytest.mark.parametrize("bad", [2, -1, np.nan, "yes"])
def test_cohort_y_true_must_be_0_or_1(cohort, bad):
    cohort["y_true"] = [1, 0, 0, bad]
    with pytest.raises(SchemaError, match="y_true"):
        validate_cohort_frame(cohort)


# "50–59" uses an en dash instead of a hyphen: it looks right but is a different string.
@pytest.mark.parametrize("bad", ["90+", "50–59", "50 - 59", "[50-60)"])
def test_unknown_age_band_is_rejected(cohort, bad):
    cohort["age_band"] = ["<50", "50-59", "70-79", bad]
    with pytest.raises(SchemaError, match="age_band"):
        validate_cohort_frame(cohort)


@pytest.mark.parametrize("bad", ["F", "female", "Unknown/Invalid"])
def test_unknown_sex_value_is_rejected(cohort, bad):
    cohort["sex"] = ["Female", "Male", "Unknown", bad]
    with pytest.raises(SchemaError, match="sex"):
        validate_cohort_frame(cohort)


@pytest.mark.parametrize("col", GROUP_COLS)
def test_cohort_nan_in_group_column_is_rejected(cohort, col):
    # Bug A: groupby would silently drop these patients.
    with pytest.raises(SchemaError, match=col):
        validate_cohort_frame(blank_first_row(cohort, col))


def test_race_is_not_restricted_to_a_fixed_set(cohort):
    # Race categories differ between datasets, so any non-missing label is allowed.
    cohort["race"] = ["Black", "White", "Asian", "Pacific Islander"]
    validate_cohort_frame(cohort)


# ---------------------------------------------------------------- audit frame


def test_valid_audit_is_returned_as_the_same_object(audit):
    assert validate_audit_frame(audit, GROUP_COLS) is audit


def test_audit_is_not_mutated(audit):
    before = audit.copy()
    validate_audit_frame(audit, GROUP_COLS)
    pd.testing.assert_frame_equal(audit, before)


def test_empty_audit_is_rejected(audit):
    with pytest.raises(SchemaError, match="empty"):
        validate_audit_frame(audit.iloc[0:0], GROUP_COLS)


@pytest.mark.parametrize("col", ["y_true", "y_score", "race"])
def test_audit_missing_required_column_is_rejected(audit, col):
    with pytest.raises(SchemaError, match=col):
        validate_audit_frame(audit.drop(columns=col), ["race"])


def test_audit_needs_at_least_one_group_column(audit):
    with pytest.raises(SchemaError, match="group_cols"):
        validate_audit_frame(audit, [])


def test_audit_only_checks_the_group_columns_it_is_given(audit):
    # No sex or age_band column at all: fine, because we only audit by race.
    validate_audit_frame(audit[["y_true", "y_score", "race"]], ["race"])


@pytest.mark.parametrize("bad", [2, np.nan, "yes"])
def test_audit_y_true_must_be_0_or_1(audit, bad):
    audit["y_true"] = [1, 0, 0, bad]
    with pytest.raises(SchemaError, match="y_true"):
        validate_audit_frame(audit, GROUP_COLS)


def test_audit_y_true_with_a_single_class_is_rejected(audit):
    # AUC is undefined unless there are both positives and negatives.
    audit["y_true"] = [0, 0, 0, 0]
    with pytest.raises(SchemaError, match="one class"):
        validate_audit_frame(audit, GROUP_COLS)


@pytest.mark.parametrize("bad", [-0.1, 1.5, np.nan])
def test_audit_y_score_must_be_a_probability(audit, bad):
    # Bug B: Brier score and calibration need scores on the 0-1 scale.
    audit["y_score"] = [0.72, 0.15, 0.5, bad]
    with pytest.raises(SchemaError, match="y_score"):
        validate_audit_frame(audit, GROUP_COLS)


def test_audit_y_score_must_be_numeric(audit):
    audit["y_score"] = ["high", "low", "low", "high"]
    with pytest.raises(SchemaError, match="y_score"):
        validate_audit_frame(audit, GROUP_COLS)


@pytest.mark.parametrize("col", GROUP_COLS)
def test_audit_nan_in_group_column_is_rejected(audit, col):
    with pytest.raises(SchemaError, match=col):
        validate_audit_frame(blank_first_row(audit, col), GROUP_COLS)


def test_small_groups_are_not_rejected(audit):
    # Every group here has 1-2 patients. Schema checks validity; metrics/ warns
    # about reliability (groups under 30), so small groups must pass here.
    validate_audit_frame(audit, GROUP_COLS)

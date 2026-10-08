"""Tests for equitypanel.model.features: turning a cohort frame into model inputs."""

from pathlib import Path

import pandas as pd
import pytest

from equitypanel.model.features import (
    CATEGORICAL_FEATURES,
    DIAG_COLS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_features,
    icd9_group,
)

REAL_UCI_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "diabetic_data.csv"
NEVER_FEATURES = [
    "patient_id",
    "y_true",
    "race",
    "sex",
    "payer_code",
    "weight",
    "medical_specialty",
    "readmitted",
]

# ---------------------------------------------------------------- icd9_group


@pytest.mark.parametrize(
    "code, group",
    [
        ("390", "circulatory"),
        ("428", "circulatory"),
        ("459", "circulatory"),
        ("459.9", "circulatory"),
        ("785", "circulatory"),
        ("460", "respiratory"),
        ("486", "respiratory"),
        ("519", "respiratory"),
        ("786", "respiratory"),
        ("520", "digestive"),
        ("579", "digestive"),
        ("787", "digestive"),
        ("250", "diabetes"),
        ("250.01", "diabetes"),
        ("250.83", "diabetes"),
        ("249", "other"),
        ("251", "other"),
        ("800", "injury"),
        ("999", "injury"),
        ("710", "musculoskeletal"),
        ("739", "musculoskeletal"),
        ("580", "genitourinary"),
        ("629", "genitourinary"),
        ("788", "genitourinary"),
        ("140", "neoplasms"),
        ("239", "neoplasms"),
        ("038", "other"),
        ("789", "other"),
        ("V57", "other"),
        ("E909", "other"),
        ("?", "missing"),
    ],
)
def test_icd9_group(code, group):
    assert icd9_group(code) == group


@pytest.mark.parametrize("code", ["", "abc", "4x8", "V", None])
def test_icd9_group_rejects_unknown_formats(code):
    with pytest.raises(ValueError, match="ICD-9"):
        icd9_group(code)


# ---------------------------------------------------------------- feature lists


def test_feature_lists():
    assert len(NUMERIC_FEATURES) == 8
    assert len(CATEGORICAL_FEATURES) == 10
    assert DIAG_COLS == ("diag_1", "diag_2", "diag_3")
    assert FEATURE_COLUMNS == NUMERIC_FEATURES + CATEGORICAL_FEATURES + DIAG_COLS
    assert "age_band" in CATEGORICAL_FEATURES
    assert "number_inpatient" in NUMERIC_FEATURES


@pytest.mark.parametrize("col", NEVER_FEATURES)
def test_protected_label_and_proxy_columns_are_never_features(col):
    assert col not in FEATURE_COLUMNS


# ---------------------------------------------------------------- build_features


def test_build_features_columns_and_rows(cohort):
    X = build_features(cohort)
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert X.index.equals(cohort.index)


def test_build_features_keeps_a_custom_index(cohort):
    shuffled = cohort.sample(frac=1, random_state=0)
    X = build_features(shuffled)
    assert X.index.equals(shuffled.index)


def test_counts_are_float_and_unchanged(cohort):
    X = build_features(cohort)
    for col in NUMERIC_FEATURES:
        assert X[col].dtype == "float64"
        assert (X[col] == cohort[col]).all()


def test_categories_are_strings_including_id_columns(cohort):
    X = build_features(cohort)
    for col in CATEGORICAL_FEATURES:
        assert X[col].map(type).eq(str).all(), col
    assert set(X["admission_type_id"]) <= {"1", "2", "3", "6"}


def test_none_and_question_mark_stay_as_categories(cohort):
    df = cohort.copy()
    df.loc[0, ["A1Cresult", "max_glu_serum", "diag_3"]] = ["None", "None", "?"]
    X = build_features(df)
    assert X.loc[0, "A1Cresult"] == "None"
    assert X.loc[0, "max_glu_serum"] == "None"
    assert X.loc[0, "diag_3"] == "missing"


def test_diagnoses_are_grouped(cohort):
    X = build_features(cohort)
    for col in DIAG_COLS:
        expected = cohort[col].map(icd9_group)
        assert (X[col] == expected).all()


def test_does_not_modify_input(cohort):
    before = cohort.copy()
    build_features(cohort)
    pd.testing.assert_frame_equal(cohort, before)


def test_missing_column_raises_naming_it(cohort):
    with pytest.raises(ValueError, match="number_inpatient"):
        build_features(cohort.drop(columns=["number_inpatient"]))


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_UCI_PATH.exists(), reason="data/raw/diabetic_data.csv not downloaded")
def test_real_uci_builds_features():
    from equitypanel.data.uci import load_uci

    cohort = load_uci(REAL_UCI_PATH)
    X = build_features(cohort)
    assert len(X) == len(cohort)
    assert X.notna().all().all()
    assert "missing" in set(X["diag_3"])
    assert set(X["diag_1"]) <= {
        "circulatory",
        "respiratory",
        "digestive",
        "diabetes",
        "injury",
        "musculoskeletal",
        "genitourinary",
        "neoplasms",
        "other",
        "missing",
    }

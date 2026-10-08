"""Tests for equitypanel.model.baseline: split, fit, score, threshold, audit frame."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from conftest import make_cohort
from equitypanel.data.schema import SchemaError, validate_audit_frame
from equitypanel.model.baseline import (
    capacity_threshold,
    fit_baseline,
    make_pipeline,
    model_weights,
    predict_risk,
    split_cohort,
    to_audit_frame,
)


@pytest.fixture(scope="module")
def big_cohort():
    return make_cohort(n=4000, seed=1)


@pytest.fixture(scope="module")
def fitted(big_cohort):
    train, test = split_cohort(big_cohort, seed=0)
    return fit_baseline(train), train, test


# ---------------------------------------------------------------- split_cohort


def test_split_sizes_and_no_overlap(cohort):
    train, test = split_cohort(cohort)
    assert len(train) == 1400 and len(test) == 600
    assert set(train["patient_id"]).isdisjoint(test["patient_id"])
    assert set(train["patient_id"]) | set(test["patient_id"]) == set(cohort["patient_id"])


def test_split_is_stratified(cohort):
    train, test = split_cohort(cohort)
    assert train["y_true"].sum() + test["y_true"].sum() == cohort["y_true"].sum()
    assert abs(train["y_true"].mean() - test["y_true"].mean()) < 0.01


def test_split_resets_index_and_keeps_columns(cohort):
    train, test = split_cohort(cohort)
    assert list(train.columns) == list(cohort.columns)
    assert train.index.equals(pd.RangeIndex(len(train)))
    assert test.index.equals(pd.RangeIndex(len(test)))


def test_split_reproducible_with_seed(cohort):
    a, _ = split_cohort(cohort, seed=0)
    b, _ = split_cohort(cohort, seed=0)
    c, _ = split_cohort(cohort, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not a["patient_id"].equals(c["patient_id"])


@pytest.mark.parametrize("test_size", [0, 1, -0.2, 1.5])
def test_split_rejects_bad_test_size(cohort, test_size):
    with pytest.raises(ValueError, match="test_size"):
        split_cohort(cohort, test_size=test_size)


def test_split_does_not_modify_input(cohort):
    before = cohort.copy()
    split_cohort(cohort)
    pd.testing.assert_frame_equal(cohort, before)


# ---------------------------------------------------------------- fit and predict


def test_pipeline_is_unfitted_logistic_regression():
    pipe = make_pipeline()
    assert type(pipe[-1]).__name__ == "LogisticRegression"
    assert pipe[-1].class_weight is None  # no reweighting: probabilities stay calibrated


def test_predictions_are_probabilities(fitted):
    model, _, test = fitted
    p = predict_risk(model, test)
    assert isinstance(p, np.ndarray)
    assert p.dtype == np.float64
    assert p.shape == (len(test),)
    assert ((p >= 0) & (p <= 1)).all()


def test_model_learns_the_signal(fitted):
    # The fake risk depends only on number_inpatient, so ranking by it is the best
    # possible model. The fitted model should come close to that, and beat chance.
    model, _, test = fitted
    model_auc = roc_auc_score(test["y_true"], predict_risk(model, test))
    best_auc = roc_auc_score(test["y_true"], test["number_inpatient"])
    assert model_auc > 0.6
    assert model_auc > best_auc - 0.03


def test_probabilities_are_roughly_calibrated(fitted):
    model, _, test = fitted
    p = predict_risk(model, test)
    assert p.mean() == pytest.approx(test["y_true"].mean(), abs=0.03)


def test_fit_is_deterministic(big_cohort, fitted):
    model, train, test = fitted
    again = fit_baseline(train)
    np.testing.assert_allclose(predict_risk(model, test), predict_risk(again, test))


def test_predictions_ignore_race_and_sex(fitted):
    model, _, test = fitted
    swapped = test.assign(race="Asian", sex="Male")
    np.testing.assert_array_equal(predict_risk(model, test), predict_risk(model, swapped))


def test_unseen_category_does_not_crash(fitted):
    model, _, test = fitted
    odd = test.head(5).assign(admission_type_id=99, A1Cresult=">9")
    p = predict_risk(model, odd)
    assert ((p >= 0) & (p <= 1)).all()


def test_fit_does_not_modify_input(big_cohort):
    train, _ = split_cohort(big_cohort)
    before = train.copy()
    fit_baseline(train)
    pd.testing.assert_frame_equal(train, before)


def test_fit_converges_without_warnings(big_cohort):
    import warnings

    train, _ = split_cohort(big_cohort)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_baseline(train)


# ---------------------------------------------------------------- model_weights


def test_model_weights_table(fitted):
    model, _, _ = fitted
    w = model_weights(model)
    assert list(w.columns) == ["feature", "weight"]
    assert w["weight"].abs().is_monotonic_decreasing
    assert w.index.equals(pd.RangeIndex(len(w)))
    assert not w["feature"].str.contains("race|sex|payer", case=False).any()
    # The planted signal should be the top weight, and positive.
    assert w.loc[0, "feature"] == "number_inpatient"
    assert w.loc[0, "weight"] > 0


# ---------------------------------------------------------------- capacity_threshold


def test_capacity_threshold_flags_the_top_share():
    scores = np.arange(1000) / 1000
    t = capacity_threshold(scores, flag_share=0.10)
    assert isinstance(t, float)
    assert np.mean(scores >= t) == pytest.approx(0.10, abs=0.002)


def test_capacity_threshold_default_is_ten_percent():
    scores = np.linspace(0, 1, 501)
    assert capacity_threshold(scores) == pytest.approx(0.9)


@pytest.mark.parametrize("share", [0, 1, -0.1, 1.2])
def test_capacity_threshold_rejects_bad_share(share):
    with pytest.raises(ValueError, match="flag_share"):
        capacity_threshold([0.1, 0.2], flag_share=share)


def test_capacity_threshold_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        capacity_threshold([])


# ---------------------------------------------------------------- to_audit_frame


def test_to_audit_frame(fitted):
    model, _, test = fitted
    p = predict_risk(model, test)
    frame = to_audit_frame(test, p)
    assert list(frame.columns) == ["y_true", "y_score", "race", "sex", "age_band"]
    assert (frame["y_score"].to_numpy() == p).all()
    assert (frame["y_true"] == test["y_true"]).all()
    validate_audit_frame(frame, ["race", "sex", "age_band"])


def test_to_audit_frame_length_mismatch(cohort):
    with pytest.raises(ValueError, match="length"):
        to_audit_frame(cohort, np.full(len(cohort) - 1, 0.1))


def test_to_audit_frame_rejects_scores_outside_0_1(cohort):
    with pytest.raises(SchemaError):
        to_audit_frame(cohort, np.full(len(cohort), 1.5))

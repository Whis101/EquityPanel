"""Baseline readmission model: logistic regression on the UCI features.

Everything is fit on the training patients only: the scaler, the one-hot
categories, the weights and the flagging threshold. The test set is scored once.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
except ImportError as err:  # pragma: no cover
    raise ImportError(
        'equitypanel.model needs scikit-learn: pip install "equitypanel[model]"'
    ) from err

from equitypanel.data.schema import validate_audit_frame
from equitypanel.model.features import (
    CATEGORICAL_FEATURES,
    DIAG_COLS,
    NUMERIC_FEATURES,
    build_features,
)

AUDIT_GROUP_COLS = ("race", "sex", "age_band")


def _check_share(value: float, name: str) -> None:
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )
    if not (is_number and 0 < value < 1):
        raise ValueError(f"{name} must be a number strictly between 0 and 1, got {value!r}")


def split_cohort(
    cohort: pd.DataFrame, *, test_size: float = 0.3, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split patients into (train, test), stratified by y_true. Indexes are reset.

    The cohort has one row per patient, so no patient lands in both parts.
    """
    _check_share(test_size, "test_size")
    train, test = train_test_split(
        cohort, test_size=test_size, stratify=cohort["y_true"], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_pipeline() -> Pipeline:
    """Unfitted pipeline: cohort frame -> features -> scale / one-hot -> logistic regression."""
    encode = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            # An unseen category at predict time becomes all zeros instead of an error.
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES + DIAG_COLS)),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features)),
            ("encode", encode),
            # No class weighting, so predicted probabilities stay calibrated.
            ("model", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )


def fit_baseline(train: pd.DataFrame) -> Pipeline:
    """Fit the baseline pipeline on the training cohort."""
    return make_pipeline().fit(train, train["y_true"])


def predict_risk(model: Pipeline, cohort: pd.DataFrame) -> np.ndarray:
    """Predicted probability of readmission within 30 days, one per patient."""
    return model.predict_proba(cohort)[:, 1].astype(np.float64)


def model_weights(model: Pipeline) -> pd.DataFrame:
    """The fitted weights as (feature, weight), largest absolute weight first.

    Counts are standardized, so their weight is per standard deviation.
    """
    names = model.named_steps["encode"].get_feature_names_out()
    weights = model.named_steps["model"].coef_[0]
    table = pd.DataFrame({"feature": names, "weight": weights})
    order = table["weight"].abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


def capacity_threshold(scores: ArrayLike, flag_share: float = 0.10) -> float:
    """Score cutoff that flags the top flag_share of patients (score >= cutoff)."""
    _check_share(flag_share, "flag_share")
    s = np.asarray(scores, dtype=float)
    if len(s) == 0:
        raise ValueError("scores is empty")
    return float(np.quantile(s, 1 - flag_share))


def to_audit_frame(cohort: pd.DataFrame, scores: ArrayLike) -> pd.DataFrame:
    """Labels, scores and group columns, ready for metrics.audit(). Validated."""
    s = np.asarray(scores, dtype=float)
    if len(s) != len(cohort):
        raise ValueError(f"scores has length {len(s)}, cohort has {len(cohort)} rows")
    frame = cohort[["y_true", *AUDIT_GROUP_COLS]].reset_index(drop=True)
    frame.insert(1, "y_score", s)
    return validate_audit_frame(frame, AUDIT_GROUP_COLS)

"""The two CKD-EPI creatinine equations for estimated GFR (eGFR), in mL/min/1.73 m².

- CKD-EPI 2009 (Levey et al., Ann Intern Med 2009): multiplies by 1.159 for Black patients.
- CKD-EPI 2021 (Inker et al., N Engl J Med 2021): no race term, coefficients re-fitted.

Both take arrays (or scalars) of serum creatinine in mg/dL, age in years and a female
flag, and return a float numpy array. Inputs are checked by _check_inputs first.
"""

import numpy as np
from numpy.typing import ArrayLike

MIN_AGE = 18  # both equations were developed on adults only


def _as_float(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        raise ValueError(f"{name} has missing values")
    return arr


def _as_bool(values: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype != bool and not np.isin(arr, [0, 1]).all():
        raise ValueError(f"{name} must be True/False (or 1/0)")
    return arr.astype(bool)


def _check_inputs(
    creatinine: ArrayLike, age: ArrayLike, female: ArrayLike, black: ArrayLike | None = None
) -> tuple[np.ndarray, ...]:
    """Validate and convert the inputs; return them as numpy arrays of one shape.

    Raises ValueError for missing values, creatinine <= 0, age under 18, non-boolean
    flags, or arrays of different lengths.
    """
    scr = _as_float(creatinine, "creatinine")
    age_arr = _as_float(age, "age")
    fem = _as_bool(female, "female")
    arrays = [scr, age_arr, fem]
    if black is not None:
        arrays.append(_as_bool(black, "black"))
    if len({a.shape for a in arrays}) > 1:
        raise ValueError(f"inputs must have the same shape, got {[a.shape for a in arrays]}")
    if (scr <= 0).any():
        raise ValueError("creatinine must be positive (mg/dL)")
    if (age_arr < MIN_AGE).any():
        raise ValueError(f"age must be at least {MIN_AGE}: the equations are for adults")
    return tuple(arrays)


def egfr_ckd_epi_2009(
    creatinine: ArrayLike, age: ArrayLike, female: ArrayLike, black: ArrayLike
) -> np.ndarray:
    """CKD-EPI 2009 creatinine equation (race-based).

    eGFR = 141 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^-1.209 × 0.993^Age
               × 1.018 [if female] × 1.159 [if Black]
    κ = 0.7 (female) or 0.9 (male); α = -0.329 (female) or -0.411 (male).
    """
    scr, age, female, black = _check_inputs(creatinine, age, female, black)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.329, -0.411)
    ratio = scr / kappa
    return (
        141
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.209
        * 0.993**age
        * np.where(female, 1.018, 1.0)
        * np.where(black, 1.159, 1.0)
    )


def egfr_ckd_epi_2021(creatinine: ArrayLike, age: ArrayLike, female: ArrayLike) -> np.ndarray:
    """CKD-EPI 2021 creatinine equation (race-free).

    eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^-1.200 × 0.9938^Age × 1.012 [if female]
    κ = 0.7 (female) or 0.9 (male); α = -0.241 (female) or -0.302 (male).
    """
    scr, age, female = _check_inputs(creatinine, age, female)
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    return (
        142
        * np.minimum(ratio, 1) ** alpha
        * np.maximum(ratio, 1) ** -1.200
        * 0.9938**age
        * np.where(female, 1.012, 1.0)
    )

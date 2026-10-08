"""Shared test helpers."""

import numpy as np
import pandas as pd
import pytest

DIAG_CODES = ["428", "250.83", "486", "V57", "?", "715", "599", "174", "820", "038"]


def make_cohort(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """A fake cohort frame shaped like clean_uci() output, with a learnable signal.

    Risk rises with number_inpatient, so a working model should beat chance clearly.
    Race and sex are drawn independently of everything else.
    """
    rng = np.random.default_rng(seed)
    inpatient = rng.poisson(0.6, n)
    logit = -3.0 + 0.9 * inpatient
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(
        {
            "patient_id": np.arange(1000, 1000 + n),
            "y_true": y,
            "race": rng.choice(
                ["White", "Black", "Hispanic", "Asian"], n, p=[0.7, 0.2, 0.06, 0.04]
            ),
            "sex": rng.choice(["Female", "Male"], n),
            "age_band": rng.choice(["<50", "50-59", "60-69", "70-79", "80+"], n),
            "weight": "?",
            "payer_code": rng.choice(["MC", "?", "HM"], n),
            "medical_specialty": rng.choice(["Cardiology", "?"], n),
            "admission_type_id": rng.choice([1, 2, 3, 6], n),
            "discharge_disposition_id": rng.choice([1, 3, 6], n),
            "admission_source_id": rng.choice([1, 7], n),
            "time_in_hospital": rng.integers(1, 14, n),
            "num_lab_procedures": rng.integers(1, 100, n),
            "num_procedures": rng.integers(0, 6, n),
            "num_medications": rng.integers(1, 60, n),
            "number_outpatient": rng.poisson(0.3, n),
            "number_emergency": rng.poisson(0.2, n),
            "number_inpatient": inpatient,
            "diag_1": rng.choice(DIAG_CODES, n),
            "diag_2": rng.choice(DIAG_CODES, n),
            "diag_3": rng.choice(DIAG_CODES, n),
            "number_diagnoses": rng.integers(1, 16, n),
            "max_glu_serum": rng.choice(["None", ">200", "Norm"], n),
            "A1Cresult": rng.choice(["None", ">8", "Norm"], n),
            "metformin": rng.choice(["No", "Steady", "Up"], n),
            "insulin": rng.choice(["No", "Steady", "Up", "Down"], n),
            "change": rng.choice(["No", "Ch"], n),
            "diabetesMed": rng.choice(["Yes", "No"], n),
        }
    )


@pytest.fixture
def cohort() -> pd.DataFrame:
    return make_cohort()

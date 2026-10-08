"""Apply both eGFR equations to a lab frame and count who changes category, by group.

A patient is reclassified when their CKD stage differs between 2009 and 2021, or when
they are on different sides of a clinical line (LINES) under the two equations. Counts,
not averages: a 3-point drop matters only if it moves someone across a line.
"""

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from equitypanel.data.schema import AGE_BANDS, validate_lab_frame
from equitypanel.reclassification.egfr import egfr_ckd_epi_2009, egfr_ckd_epi_2021

# KDIGO stages, healthiest first, with the lowest eGFR each one includes.
STAGES = ("G1", "G2", "G3a", "G3b", "G4", "G5")
STAGE_LOWER_BOUNDS = (90.0, 60.0, 45.0, 30.0, 15.0, -np.inf)

# Clinical lines: name -> (eGFR cutoff, whether the cutoff itself counts as below).
LINES = {
    "ckd_lt60": (60.0, False),  # eGFR < 60: chronic kidney disease
    "referral_lt30": (30.0, False),  # eGFR < 30: nephrologist referral, no metformin
    "waitlist_le20": (20.0, True),  # eGFR <= 20: may start accruing transplant waiting time
}

# Creatinine values outside this range (mg/dL) are treated as data errors and dropped.
CREATININE_RANGE = (0.2, 20.0)

SMALL_GROUP_N = 30
RACE_TERM_COL = "race_term"  # "Black" or "non-Black": who got the 2009 multiplier
DEFAULT_GROUP_COLS = (RACE_TERM_COL, "race", "sex", "age_band")
DIRECTIONS = ("lower", "higher", "same")


def ckd_stage(egfr: pd.Series | np.ndarray) -> np.ndarray:
    """Map eGFR values to KDIGO stage labels (G1-G5). Exactly 60.0 is G2, exactly 15.0 is G4."""
    values = np.asarray(egfr, dtype=float)
    if np.isnan(values).any():
        raise ValueError("egfr has missing values")
    # Count how many stage floors the value is below: 0 floors = G1, 5 = G5.
    stage_idx = np.zeros(values.shape, dtype=int)
    for bound in STAGE_LOWER_BOUNDS[:-1]:
        stage_idx += values < bound
    return np.asarray(STAGES, dtype=object)[stage_idx]


def below_line(egfr: pd.Series | np.ndarray, line: str) -> np.ndarray:
    """True where eGFR is on the 'sick' side of LINES[line]."""
    cutoff, inclusive = LINES[line]
    values = np.asarray(egfr, dtype=float)
    return values <= cutoff if inclusive else values < cutoff


def filter_plausible(lab: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows whose creatinine is outside CREATININE_RANGE. Returns (kept, n_dropped)."""
    low, high = CREATININE_RANGE
    keep = lab["creatinine_mg_dl"].between(low, high)
    return lab[keep].reset_index(drop=True), int((~keep).sum())


def reclassify(lab: pd.DataFrame) -> pd.DataFrame:
    """One row per patient: both eGFRs, both stages, direction of change, and every line.

    Columns: patient_id, race_term, race, sex, age_band, age, creatinine_mg_dl,
    egfr_2009, egfr_2021, stage_2009, stage_2021, direction, then <line>_2009 and
    <line>_2021 for each line. direction is "lower" when the 2021 stage is worse
    (lower eGFR category) than 2009, "higher" when better, "same" otherwise.
    Raises ValueError if any sex is not Female or Male: both equations need it.
    """
    validate_lab_frame(lab)
    bad_sex = sorted(set(lab["sex"]) - {"Female", "Male"})
    if bad_sex:
        raise ValueError(f"sex must be Female or Male for the eGFR equations, found {bad_sex}")

    female = (lab["sex"] == "Female").to_numpy()
    black = (lab["race"] == "Black").to_numpy()
    scr = lab["creatinine_mg_dl"].to_numpy()
    age = lab["age"].to_numpy()

    out = pd.DataFrame(
        {
            "patient_id": lab["patient_id"].to_numpy(),
            RACE_TERM_COL: np.where(black, "Black", "non-Black"),
            "race": lab["race"].to_numpy(),
            "sex": lab["sex"].to_numpy(),
            "age_band": lab["age_band"].to_numpy(),
            "age": age,
            "creatinine_mg_dl": scr,
            "egfr_2009": egfr_ckd_epi_2009(scr, age, female, black),
            "egfr_2021": egfr_ckd_epi_2021(scr, age, female),
        }
    )
    out["stage_2009"] = ckd_stage(out["egfr_2009"])
    out["stage_2021"] = ckd_stage(out["egfr_2021"])
    rank = {stage: i for i, stage in enumerate(STAGES)}
    diff = out["stage_2021"].map(rank) - out["stage_2009"].map(rank)
    out["direction"] = np.select([diff > 0, diff < 0], ["lower", "higher"], default="same")
    for line in LINES:
        out[f"{line}_2009"] = below_line(out["egfr_2009"], line)
        out[f"{line}_2021"] = below_line(out["egfr_2021"], line)
    return out


def _group_order(col: str, values: pd.Series) -> list[str]:
    present = set(values)
    if col == "age_band":
        return [band for band in AGE_BANDS if band in present]
    return sorted(present)


def _summary_row(col: str, group: str, part: pd.DataFrame) -> dict:
    n = len(part)
    counts = part["direction"].value_counts()
    row = {"group_col": col, "group": group, "n": n, "small_group": n < SMALL_GROUP_N}
    for d in DIRECTIONS:
        row[f"n_{d}"] = int(counts.get(d, 0))
    for d in DIRECTIONS:
        row[f"pct_{d}"] = row[f"n_{d}"] / n
    for line in LINES:
        before = part[f"{line}_2009"]
        after = part[f"{line}_2021"]
        row[f"{line}_2009"] = int(before.sum())
        row[f"{line}_2021"] = int(after.sum())
        row[f"{line}_newly_below"] = int((after & ~before).sum())
        row[f"{line}_newly_above"] = int((before & ~after).sum())
    return row


def reclassification_summary(
    df: pd.DataFrame, group_cols: Sequence[str] = DEFAULT_GROUP_COLS
) -> pd.DataFrame:
    """Per-group counts from a reclassify() frame, plus an "all" row first.

    One row per (group_col, group): n, small_group (n < 30), n_/pct_ lower/higher/same,
    and for each line: <line>_2009, <line>_2021 (number below it), <line>_newly_below
    (below under 2021 only) and <line>_newly_above (below under 2009 only).
    Age bands keep their natural order; other groups are sorted alphabetically.
    """
    if df.empty:
        raise ValueError("df is empty")
    missing = [col for col in group_cols if col not in df.columns]
    if missing:
        raise ValueError(f"group column(s) not in df: {missing}")
    rows = [_summary_row("all", "All", df)]
    for col in group_cols:
        for group in _group_order(col, df[col]):
            rows.append(_summary_row(col, group, df[df[col] == group]))
    return pd.DataFrame(rows)


def stage_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """6 × 6 counts: rows = stage under 2009, columns = stage under 2021 (all stages shown)."""
    table = pd.crosstab(
        pd.Categorical(df["stage_2009"], categories=STAGES),
        pd.Categorical(df["stage_2021"], categories=STAGES),
        dropna=False,
    )
    table.index = pd.Index(STAGES, name="stage_2009")
    table.columns = pd.Index(STAGES, name="stage_2021")
    return table


def headline_numbers(summary: pd.DataFrame) -> dict:
    """The race_term rows of a summary as {"Black": {...}, "non-Black": {...}} for JSON."""
    rows = summary[summary["group_col"] == RACE_TERM_COL]
    if set(rows["group"]) != {"Black", "non-Black"}:
        raise ValueError("summary needs race_term rows for both Black and non-Black patients")
    # to_json turns numpy ints, floats and bools into plain JSON values.
    records = json.loads(rows.drop(columns="group_col").to_json(orient="records"))
    return {record.pop("group"): record for record in records}

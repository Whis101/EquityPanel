"""End-to-end: filter a lab frame, apply both equations, summarise by group, write files."""

import json
from os import PathLike
from pathlib import Path

import pandas as pd

from equitypanel.reclassification.plot import SYNTHETIC_LABEL, headline_chart
from equitypanel.reclassification.reclassify import (
    CREATININE_RANGE,
    filter_plausible,
    headline_numbers,
    reclassification_summary,
    reclassify,
    stage_crosstab,
)

OUTPUT_FILES = (
    "egfr_patients.csv",
    "egfr_summary.csv",
    "egfr_crosstab.csv",
    "egfr_headline.json",
    "egfr_headline.png",
)


def run_reclassification(lab: pd.DataFrame, out_dir: str | PathLike) -> dict:
    """Drop implausible creatinine, reclassify everyone, write OUTPUT_FILES to out_dir.

    Everything is computed before anything is written, so a failure leaves no
    partial output. Returns the headline dict that is also saved as egfr_headline.json.
    """
    kept, n_dropped = filter_plausible(lab)
    patients = reclassify(kept)
    summary = reclassification_summary(patients)
    crosstab = stage_crosstab(patients)
    fig = headline_chart(summary)
    headline = {
        "data_label": SYNTHETIC_LABEL,
        "n_input": len(lab),
        "n_dropped_implausible": n_dropped,
        "creatinine_range_mg_dl": list(CREATININE_RANGE),
        "n_patients": len(patients),
        "by_race_term": headline_numbers(summary),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    patients.to_csv(out / "egfr_patients.csv", index=False)
    summary.to_csv(out / "egfr_summary.csv", index=False)
    crosstab.to_csv(out / "egfr_crosstab.csv")
    (out / "egfr_headline.json").write_text(json.dumps(headline, indent=2) + "\n")
    fig.savefig(out / "egfr_headline.png")
    return headline

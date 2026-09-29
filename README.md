# EquityPanel

A Python package for auditing clinical risk models for performance gaps across
demographic groups (race, sex, age band).

> **Work in progress.** Milestone 1 (scaffold and data layer) is under way.
> Most of the features below are planned, not built yet. See [Status](#status).

## Goal

Given a dataset of true outcomes, a model's predicted risk scores and patient
group columns, EquityPanel will report how well the model performs for each
group, so that gaps between groups are visible instead of hidden inside an
overall accuracy number. Planned pieces:

- **Subgroup metrics:** per-group AUC, calibration, error rates and bootstrap
  confidence intervals, with small groups flagged.
- **Threshold analysis:** how flag rates and error rates change per group as
  the risk cutoff moves.
- **eGFR reclassification:** how many patients change kidney-disease stage when
  the race coefficient is removed (CKD-EPI 2009 vs 2021 equations).
- **Report:** a self-contained HTML report of the results.

## Status

| Area | State |
|---|---|
| Package scaffold, tooling (pytest, Ruff), MIT license | Done |
| Schema validation for cohort and audit frames (`equitypanel.data.schema`) | Done, with tests |
| Dataset loaders | Next |
| Metrics, thresholds, reclassification, report | Planned |

## Data

The first dataset is the UCI
[Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
dataset: 101,766 hospital encounters, with 30-day readmission as the outcome.
Data files are not committed to this repository. Download the dataset yourself
and place it under `data/raw/`, which is gitignored.

## Development

Requires Python 3.11+.

    python -m venv .venv
    source .venv/Scripts/activate   # Windows Git Bash; on macOS/Linux: source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
    ruff check .

## License

MIT. See [LICENSE](LICENSE).

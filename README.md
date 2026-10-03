# EquityPanel

[![CI](https://github.com/Whis101/EquityPanel/actions/workflows/ci.yml/badge.svg)](https://github.com/Whis101/EquityPanel/actions/workflows/ci.yml)

A Python package for auditing clinical risk models for performance gaps across
demographic groups (race, sex, age band).

> **Work in progress.** Milestone 1 (scaffold and data layer) is done.
> Most of the features below are planned, not built yet. See [Status](#status).

> **Not a clinical tool.** EquityPanel is an auditing aid for research and
> education. Run it only on public or synthetic data, never on real patient
> data, and do not use its output to make decisions about individual patients.

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
| Schema validation for cohort, audit and lab frames (`equitypanel.data.schema`) | Done, with tests |
| UCI diabetes loader (`equitypanel.data.uci`) | Done, with tests |
| Synthea loader for creatinine (`equitypanel.data.synthea`) | Done, with tests |
| Subgroup metrics (`equitypanel.metrics`) | Next |
| Thresholds, reclassification, report | Planned |

## Data

Data files are not committed to this repository. Download them yourself into
`data/raw/`, which is gitignored.

- **UCI
  [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008):**
  101,766 hospital encounters, with 30-day readmission as the outcome. Put
  `diabetic_data.csv` in `data/raw/`.
- **Synthea synthetic patients:** the
  [10k CSV export](https://github.com/synthetichealth/synthea-sample-data/tree/main/downloads)
  (`10k_synthea_covid19_csv.zip`), used only for serum creatinine in the eGFR
  comparison. Put `patients.csv` and `observations.csv` in `data/raw/synthea/`.
  These patients are simulated, so any result built on them is synthetic.

## Development

Requires Python 3.11+.

    python -m venv .venv
    source .venv/Scripts/activate   # Windows Git Bash; on macOS/Linux: source .venv/bin/activate
    pip install -e ".[dev]"
    pytest
    ruff check .
    ruff format --check .

## License

MIT. See [LICENSE](LICENSE).

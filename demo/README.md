# EquityPanel interactive demo

A Streamlit app on bundled public and synthetic data: per-group audit with an adjustable
flagging cutoff, a threshold explorer, the eGFR 2009-vs-2021 result (synthetic), the saved
AI summary, and a box to try EquityPanel's number check on any sentence. No uploads and no
AI calls.

Run locally (Git Bash, repo root):

```bash
pip install -e ".[demo]"
streamlit run demo/app.py
```

**Research and education only; not a clinical tool.** See the main [README](../README.md)
for the full disclaimers.

## Data in `data/`

- `uci_test_scores.csv`: outcome, risk score from EquityPanel's baseline model, race, sex and
  age band for the 20,997 held-out patients (no patient IDs). Derived from: Clore J, Cios K,
  DeShazo J, Strack B (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C5230J. Licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); modified (columns selected, model
  scores added) by EquityPanel.
- `egfr_summary.json`, `egfr_headline.png`: aggregate results on synthetic Synthea patients
  (The MITRE Corporation; Walonoski J, et al. JAMIA 2018).
- `uci_facts.json`, `uci_summary.json`, `uci_model.json`: the aggregate facts, the saved
  checked AI summary and the model summary from the full report.

Rebuild them with `python scripts/build_demo_data.py`. Never put real patient data here: this
folder is committed and published.

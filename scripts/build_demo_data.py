"""Copy the public results the Streamlit demo needs from outputs/ into demo/data/.

Git Bash, from the repo root (after audit_uci.py, egfr_synthea.py and build_report.py --llm):
    python scripts/build_demo_data.py
Writes only public UCI test-set scores (no patient IDs) and aggregate JSON. Never point this
at real patient data: demo/data/ is committed and published.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCORE_COLUMNS = ["y_true", "y_score", "race", "sex", "age_band"]


def main() -> None:
    out, data = ROOT / "outputs", ROOT / "demo" / "data"
    data.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(out / "uci_scores.csv", dtype={"age_band": str})
    scores[SCORE_COLUMNS].round({"y_score": 6}).to_csv(data / "uci_test_scores.csv", index=False)

    egfr = pd.read_csv(out / "egfr_summary.csv", dtype={"group": str})
    (data / "egfr_summary.json").write_text(
        egfr.to_json(orient="records", indent=1) + "\n", encoding="utf-8"
    )
    for name in ("uci_facts.json", "uci_summary.json", "uci_model.json"):
        shutil.copyfile(out / name, data / name)
    shutil.copyfile(out / "egfr_headline.png", data / "egfr_headline.png")
    print(f"wrote {data}: {len(scores):,} test-set rows + aggregates")
    json.loads((data / "uci_summary.json").read_text(encoding="utf-8"))  # sanity check


if __name__ == "__main__":
    main()

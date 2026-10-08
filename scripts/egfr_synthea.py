"""Compare CKD-EPI 2009 (race-based) and 2021 (race-free) eGFR on Synthea adults.

Git Bash, from the repo root:
    python scripts/egfr_synthea.py
Writes outputs/ (gitignored). Needs data/raw/synthea/ and the [report] extra.
The patients are synthetic: label every result built from these files as such.
"""

import argparse
from pathlib import Path

from equitypanel.data.synthea import load_synthea
from equitypanel.reclassification.run import run_reclassification

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "synthea")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)

    headline = run_reclassification(load_synthea(args.data), args.out)
    print(headline["data_label"])
    print(
        f"{headline['n_patients']:,} adults "
        f"({headline['n_dropped_implausible']} dropped as implausible creatinine)"
    )
    for group, row in headline["by_race_term"].items():
        print(
            f"{group:>9}: n {row['n']:,}, stage changed {row['n_lower'] + row['n_higher']:,}, "
            f"eGFR <= 20: {row['waitlist_le20_2009']:,} (2009) -> "
            f"{row['waitlist_le20_2021']:,} (2021)"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

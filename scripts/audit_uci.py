"""Train the baseline readmission model on UCI and audit it by race, sex and age band.

Git Bash, from the repo root:
    python scripts/audit_uci.py
Writes outputs/ (gitignored). Needs data/raw/diabetic_data.csv and the [model] extra.
"""

import argparse
from pathlib import Path

from equitypanel.data.uci import load_uci
from equitypanel.model.run import run_audit

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "raw" / "diabetic_data.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--flag-share", type=float, default=0.10)
    args = parser.parse_args(argv)

    summary = run_audit(
        load_uci(args.data),
        args.out,
        n_boot=args.n_boot,
        seed=args.seed,
        flag_share=args.flag_share,
    )
    print(f"train {summary['n_train']:,} / test {summary['n_test']:,} patients")
    print(f"test AUC {summary['auc_test']:.3f}, prevalence {summary['prevalence_test']:.1%}")
    print(f"threshold {summary['threshold']:.3f} flags {summary['flag_share_test']:.1%} of test")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

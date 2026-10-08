"""Build the HTML audit report from outputs/ (UCI audit, plus the eGFR result if present).

Git Bash, from the repo root:
    python scripts/build_report.py            # report without the LLM summary
    python scripts/build_report.py --llm      # also ask Claude for a checked summary
    python scripts/build_report.py --site     # also copy the report to site/index.html
Needs the [report] extra; --llm needs the [llm] extra and ANTHROPIC_API_KEY (in .env).
Run scripts/audit_uci.py (and scripts/egfr_synthea.py) first.
"""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from equitypanel.report.facts import build_facts
from equitypanel.report.html import build_report
from equitypanel.report.summary import Summary, make_client, summarise

ROOT = Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> pd.DataFrame:
    # Group labels like "<50" must stay strings.
    return pd.read_csv(path, dtype={"group": str})


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--llm", action="store_true", help="ask Claude for a checked summary")
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="reuse outputs/uci_summary.json instead of calling the API again",
    )
    parser.add_argument("--site", action="store_true", help="also write site/index.html")
    parser.add_argument("--n-boot", type=int, default=1000, help="as used in audit_uci.py")
    parser.add_argument("--seed", type=int, default=0, help="as used in audit_uci.py")
    args = parser.parse_args(argv)
    out = args.outputs

    metrics = _read_csv(out / "uci_audit_metrics.csv")
    calibration = _read_csv(out / "uci_audit_calibration.csv")
    model_info = json.loads((out / "uci_model.json").read_text(encoding="utf-8"))
    egfr_path = out / "egfr_summary.csv"
    egfr = _read_csv(egfr_path) if egfr_path.exists() else None

    facts = build_facts(metrics, model_info, egfr)
    (out / "uci_facts.json").write_text(json.dumps(facts, indent=1) + "\n", encoding="utf-8")

    summary = None
    summary_path = out / "uci_summary.json"
    if args.reuse_summary:
        summary = Summary.from_dict(json.loads(summary_path.read_text(encoding="utf-8")))
    elif args.llm:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        client = make_client()
        if client is None:
            print("No ANTHROPIC_API_KEY found (set it in .env); building without a summary.")
        else:
            summary = summarise(facts, client)
            summary_path.write_text(
                json.dumps(summary.to_dict(), indent=1) + "\n", encoding="utf-8"
            )
            print(
                f"summary: {len(summary.findings)} findings kept, "
                f"{len(summary.rejected)} rejected, {summary.attempts} attempt(s)"
                + (f", error: {summary.error}" if summary.error else "")
            )

    html = build_report(
        metrics,
        calibration,
        model_info,
        egfr,
        summary,
        methods={"n_boot": args.n_boot, "seed": args.seed},
    )
    report_path = out / "uci_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"wrote {report_path} ({len(html) / 1024:.0f} KB)")
    if args.site:
        site = ROOT / "site"
        site.mkdir(exist_ok=True)
        shutil.copyfile(report_path, site / "index.html")
        print(f"copied to {site / 'index.html'}")


if __name__ == "__main__":
    main()

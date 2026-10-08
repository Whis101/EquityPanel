"""Tests for equitypanel.report: facts JSON, the number check, the LLM call (fake client), HTML.

No test calls the real API. FakeClient stands in for anthropic.Anthropic and returns
canned findings, including invented numbers, to prove the guardrail rejects them.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from equitypanel.metrics.audit import audit
from equitypanel.reclassification.reclassify import reclassification_summary, reclassify
from equitypanel.report.charts import calibration_chart, group_ci_chart
from equitypanel.report.facts import build_facts, reference_groups
from equitypanel.report.html import build_report, fmt, order_groups
from equitypanel.report.summary import (
    FALLBACK_BETA,
    MODEL,
    Summary,
    check_finding,
    extract_numbers,
    make_client,
    summarise,
)
from test_reclassify import make_lab

MODEL_INFO = {
    "n_train": 1400,
    "n_test": 600,
    "prevalence_test": 0.12,
    "auc_test": 0.7123,
    "threshold": 0.3,
    "flag_share_test": 0.1,
    "top_weights": [{"feature": "number_inpatient", "weight": 0.42}],
}


def make_audit_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.3, n)
    score = np.clip(0.3 + 0.25 * (y - 0.3) + rng.normal(0, 0.15, n), 0.001, 0.999)
    return pd.DataFrame(
        {
            "y_true": y,
            "y_score": score,
            "race": rng.choice(["White", "Black", "Asian"], n, p=[0.6, 0.35, 0.05]),
            "sex": rng.choice(["Female", "Male"], n),
            "age_band": rng.choice(["<50", "50-59", "60-69", "70-79", "80+"], n),
        }
    )


@pytest.fixture(scope="module")
def audited():
    report = audit(make_audit_frame(), ["race", "sex", "age_band"], 0.4, n_boot=30, seed=0)
    return report.metrics, report.calibration


@pytest.fixture(scope="module")
def egfr_summary():
    return reclassification_summary(reclassify(make_lab()))


@pytest.fixture(scope="module")
def facts_json(audited, egfr_summary):
    metrics, _ = audited
    return build_facts(metrics, MODEL_INFO, egfr_summary)


# ---------------------------------------------------------------- facts


def test_reference_groups_are_the_groups_without_gaps(audited):
    metrics, _ = audited
    refs = reference_groups(metrics)
    assert refs["race"] == "White"
    assert set(refs) == {"race", "sex", "age_band"}


def test_facts_cover_every_metric_row(audited, facts_json):
    metrics, _ = audited
    facts = facts_json["facts"]
    for _, row in metrics.iterrows():
        assert f"{row['group_col']}.{row['group']}.{row['metric']}" in facts


def test_fact_shape(audited, facts_json):
    metrics, _ = audited
    row = metrics[(metrics["group"] == "Black") & (metrics["metric"] == "fnr")].iloc[0]
    fact = facts_json["facts"]["race.Black.fnr"]
    assert fact["value"] == pytest.approx(row["value"], abs=1e-4)
    assert fact["n"] == row["n"]
    assert fact["ci_95"] == [
        pytest.approx(row["ci_low"], abs=1e-4),
        pytest.approx(row["ci_high"], abs=1e-4),
    ]
    assert "false negative" in fact["label"]
    assert "White" in facts_json["facts"]["race.Black.fnr_gap"]["label"]


def test_facts_model_and_egfr(facts_json):
    facts = facts_json["facts"]
    assert facts["model.auc_test"]["value"] == 0.7123
    assert facts["egfr.Black.waitlist_le20_2021"]["value"] == 1
    assert facts["egfr.All.n"]["value"] == 6
    assert "SYNTHETIC" in facts_json["context"]["egfr"]


def test_facts_hold_no_patient_level_data(facts_json):
    text = json.dumps(facts_json)
    assert "patient_id" not in text and "y_score" not in text
    # Every fact is a small dict of numbers and labels.
    for fact in facts_json["facts"].values():
        assert set(fact) <= {"label", "value", "n", "small_group", "ci_95"}


def test_facts_missing_values_are_null(audited):
    metrics, _ = audited
    broken = metrics.copy()
    broken.loc[broken.index[3], "value"] = np.nan
    facts = build_facts(broken)["facts"]
    row = broken.iloc[3]
    assert facts[f"{row['group_col']}.{row['group']}.{row['metric']}"]["value"] is None
    json.dumps(facts)  # NaN would not be valid JSON


def test_facts_without_model_or_egfr(audited):
    metrics, _ = audited
    got = build_facts(metrics)
    assert not any(k.startswith(("model.", "egfr.")) for k in got["facts"])
    assert "egfr" not in got["context"]


# ---------------------------------------------------------------- the number check

FACTS = {
    "race.Black.fnr": {"value": 0.7959, "n": 3826, "ci_95": [0.7522, 0.8367]},
    "race.White.fnr": {"value": 0.7769, "n": 15614, "ci_95": [0.7537, 0.7969]},
    "race.Black.fnr_gap": {"value": 0.019, "n": 3826},
    "age_band.50-59.auc": {"value": 0.663, "n": 3701},
    "egfr.Black.waitlist_le20_2009": {"value": 60},
    "egfr.Black.waitlist_le20_2021": {"value": 89},
    "egfr.non-Black.waitlist_le20_newly_above": {"value": 231},
}


def test_extract_numbers():
    got = extract_numbers("79.6% of 3,826 (CI 75.2%–83.7%), gap −0.019, 2009 → 2021, n=12")
    assert [(raw, value) for raw, value, *_ in got] == [
        ("79.6%", 79.6),
        ("3,826", 3826.0),
        ("75.2%", 75.2),
        ("83.7%", 83.7),
        ("−0.019", 0.019),
        ("2009", 2009.0),
        ("2021", 2021.0),
        ("12", 12.0),
    ]
    assert got[0][2] is True and got[0][3] == 1  # percentage, 1 decimal


@pytest.mark.parametrize(
    "text",
    [
        "Black patients' false negative rate is 79.6% (95% CI 75.2%–83.7%), n = 3,826.",
        "The false negative rate for Black patients is 0.796.",
        "About 80% of readmitted Black patients are missed.",
        "That is 1.9 points higher than for White patients (77.7%).",
        "Under the race-free 2021 formula, 89 synthetic Black patients reach eGFR 20 or below, "
        "up from 60 under 2009.",
    ],
)
def test_grounded_findings_pass(text):
    ids = [
        "race.Black.fnr",
        "race.White.fnr",
        "race.Black.fnr_gap",
        "egfr.Black.waitlist_le20_2009",
        "egfr.Black.waitlist_le20_2021",
    ]
    assert check_finding(text, ids, FACTS) is None


def test_invented_number_is_rejected():
    reason = check_finding("Black patients are missed 91% of the time.", ["race.Black.fnr"], FACTS)
    assert reason == "91% does not match any cited fact"


def test_number_from_an_uncited_fact_is_rejected():
    # 77.7% is real, but it belongs to a fact the finding didn't cite.
    reason = check_finding("White patients: 77.7%.", ["race.Black.fnr"], FACTS)
    assert reason is not None and "77.7%" in reason


def test_too_precise_rounding_is_rejected():
    # 79.60% claims more precision than 79.59%; 79.5% is just wrong.
    assert check_finding("FNR 79.5%.", ["race.Black.fnr"], FACTS) is not None


def test_computed_numbers_are_rejected():
    # 89 - 60 = 29 is arithmetic the model did itself: not a fact.
    reason = check_finding(
        "29 more Black patients reach the line.",
        ["egfr.Black.waitlist_le20_2009", "egfr.Black.waitlist_le20_2021"],
        FACTS,
    )
    assert reason == "29 does not match any cited fact"


def test_unknown_or_missing_fact_ids_are_rejected():
    assert check_finding("x", [], FACTS) == "cites no facts"
    assert "unknown" in check_finding("FNR 79.6%.", ["race.Martian.fnr"], FACTS)


def test_context_numbers_and_id_numbers_are_allowed():
    text = "For ages 50-59 the AUC is 0.663; eGFR lines are 60, 30 and 20 (2009 vs 2021)."
    assert check_finding(text, ["age_band.50-59.auc"], FACTS) is None
    # ...but not as invented percentages.
    assert check_finding("AUC 0.663, 60% of them.", ["age_band.50-59.auc"], FACTS) is not None


def test_signed_numbers_match_by_size():
    text = "231 fewer non-Black patients (−231) are at eGFR 20 or below."
    assert check_finding(text, ["egfr.non-Black.waitlist_le20_newly_above"], FACTS) is None


# ---------------------------------------------------------------- summarise (fake client)


class FakeClient:
    """Stands in for anthropic.Anthropic: returns the canned answers in order."""

    def __init__(self, *answers, stop_reason="end_turn"):
        self.answers = list(answers)
        self.calls = []
        self.stop_reason = stop_reason
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        text = answer if isinstance(answer, str) else json.dumps({"findings": answer})
        return SimpleNamespace(
            model=kwargs["model"],
            stop_reason=self.stop_reason,
            content=[SimpleNamespace(type="text", text=text)],
        )


GOOD = {"text": "Black patients' false negative rate is 79.6%.", "fact_ids": ["race.Black.fnr"]}
INVENTED = {"text": "Black patients are missed 91% of the time.", "fact_ids": ["race.Black.fnr"]}
FACTS_JSON = {"context": {}, "facts": FACTS}


def test_summarise_keeps_grounded_findings():
    client = FakeClient([GOOD])
    summary = summarise(FACTS_JSON, client)
    assert [f.text for f in summary.findings] == [GOOD["text"]]
    assert summary.rejected == [] and summary.attempts == 1 and summary.error is None


def test_summarise_rejects_invented_number_then_retries_once():
    client = FakeClient([GOOD, INVENTED], [GOOD, INVENTED])
    summary = summarise(FACTS_JSON, client)
    assert summary.attempts == 2
    assert [f.text for f in summary.findings] == [GOOD["text"]]
    # Rejections from both attempts are kept, tagged with the attempt that produced them.
    assert [(r.attempt, r.reason) for r in summary.rejected] == [
        (1, "91% does not match any cited fact"),
        (2, "91% does not match any cited fact"),
    ]
    assert [r.attempt for r in summary.rejected_final] == [2]
    assert [r.attempt for r in summary.rejected_earlier] == [1]
    # The retry tells the model what failed, after its own previous answer.
    feedback = client.calls[1]["messages"]
    assert feedback[1]["role"] == "assistant"
    assert "91% does not match" in feedback[2]["content"]


def test_summarise_retry_can_fix_the_finding():
    client = FakeClient([INVENTED], [GOOD])
    summary = summarise(FACTS_JSON, client)
    assert summary.attempts == 2
    assert len(summary.findings) == 1
    # Nothing is missing from the final summary, but the caught draft is kept as evidence.
    assert summary.rejected_final == []
    assert [(r.attempt, r.text) for r in summary.rejected_earlier] == [(1, INVENTED["text"])]


def test_summary_from_old_json_without_attempt():
    old = {
        "model": MODEL,
        "attempts": 1,
        "error": None,
        "findings": [],
        "rejected": [{"text": "x", "reason": "y"}],
    }
    assert Summary.from_dict(old).rejected_final[0].attempt == 1


def test_report_shows_caught_drafts(audited):
    metrics, calibration = audited
    summary = summarise(FACTS_JSON, FakeClient([INVENTED], [GOOD]))
    html = build_report(metrics, calibration, summary=summary)
    assert "caught 1 finding(s) in an earlier draft" in " ".join(html.split())
    assert "Attempt 1: 91% does not match any cited fact" in html
    assert "finding(s) rejected</strong>" not in html  # nothing missing from the final summary


def test_summarise_sends_only_the_facts_json_and_the_right_options():
    client = FakeClient([GOOD])
    summarise(FACTS_JSON, client)
    call = client.calls[0]
    assert call["model"] == MODEL == "claude-opus-5-5"
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "medium"
    (message,) = call["messages"]
    payload = json.loads(message["content"].split("\n", 1)[1])
    assert payload == FACTS_JSON


def test_summarise_handles_refusal_and_bad_json():
    refused = summarise(FACTS_JSON, FakeClient([GOOD], stop_reason="refusal"))
    assert refused.error == "the model declined to write a summary" and not refused.findings
    garbled = summarise(FACTS_JSON, FakeClient("not json"))
    assert garbled.error == "the model's answer was not valid JSON"


def test_summarise_caps_findings():
    client = FakeClient([GOOD] * 12)
    assert len(summarise(FACTS_JSON, client).findings) == 8


def test_summary_round_trips_through_json():
    summary = summarise(FACTS_JSON, FakeClient([GOOD, INVENTED], [GOOD, INVENTED]))
    again = Summary.from_dict(json.loads(json.dumps(summary.to_dict())))
    assert again == summary


def test_no_api_key_means_no_client(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert make_client() is None


# ---------------------------------------------------------------- charts


def test_charts_are_inline_svg(audited):
    metrics, calibration = audited
    for svg in (group_ci_chart(metrics, "race", "auc"), calibration_chart(calibration, "race")):
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert "<?xml" not in svg and "<metadata>" not in svg


def test_ci_chart_marks_small_groups(audited):
    metrics, _ = audited
    assert "Asian (small)" in group_ci_chart(metrics, "race", "fnr")


def test_chart_rejects_missing_rows(audited):
    metrics, calibration = audited
    with pytest.raises(ValueError):
        group_ci_chart(metrics, "income", "auc")
    with pytest.raises(ValueError):
        calibration_chart(calibration, "income")


# ---------------------------------------------------------------- html


def test_fmt():
    assert fmt("fnr", 0.7959) == "79.6%"
    assert fmt("fnr_gap", 0.019) == "+1.9 pts"  # percentage points, not percent
    assert fmt("fpr_gap", -0.032) == "-3.2 pts"
    assert fmt("auc", 0.6543) == "0.654"
    assert fmt("auc_gap", -0.0123) == "-0.012"
    assert fmt("n", 3826.0) == "3,826"
    assert fmt("auc", np.nan) == "n/a"


def test_order_groups_puts_age_bands_in_order(audited):
    metrics, _ = audited
    ordered = order_groups(metrics)
    bands = list(dict.fromkeys(ordered.loc[ordered["group_col"] == "age_band", "group"]))
    assert bands == ["<50", "50-59", "60-69", "70-79", "80+"]


def test_report_without_summary_or_egfr(audited):
    metrics, calibration = audited
    html = build_report(metrics, calibration)
    assert html.startswith("<!doctype html>")
    assert "Summary not generated" in html
    assert "Auditing aid for research and education, not a clinical" in html
    assert "eGFR" not in html
    assert html.count("<svg") == 9  # 3 group columns x (AUC, FNR, calibration)
    assert "<script" not in html  # no JavaScript, no external requests
    assert "http://" not in html.replace("http://www.w3.org", "")


def test_report_with_everything(audited, egfr_summary):
    metrics, calibration = audited
    summary = summarise(FACTS_JSON, FakeClient([GOOD, INVENTED], [GOOD, INVENTED]))
    html = build_report(
        metrics, calibration, MODEL_INFO, egfr_summary, summary, methods={"n_boot": 30, "seed": 0}
    )
    assert "Synthetic patients (Synthea). Not real-world rates." in html
    assert GOOD["text"].replace("'", "&#39;") in html
    assert "1 finding(s) rejected</strong>" in html  # final attempt
    assert "caught 1 finding(s)" in html  # attempt 1, rewritten
    assert "91% does not match any cited fact" in html
    assert "0.712" in html and "20,997" not in html
    assert "30 resamples, seed 0" in html
    assert "number_inpatient" in html


def test_report_escapes_group_names(audited):
    metrics, calibration = audited
    evil = "<script>alert(1)</script>"
    metrics = metrics.replace({"group": {"Asian": evil}})
    calibration = calibration.replace({"group": {"Asian": evil}})
    html = build_report(metrics, calibration)
    assert evil not in html
    assert "&lt;script&gt;" in html


def test_report_escapes_llm_text(audited):
    metrics, calibration = audited
    bad = Summary(findings=[], rejected=[], error="<img src=x onerror=alert(1)>")
    html = build_report(metrics, calibration, summary=bad)
    assert "<img src=x" not in html


# ---------------------------------------------------------------- scripts/build_report.py


def test_build_report_script(audited, egfr_summary, tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path

    metrics, calibration = audited
    metrics.to_csv(tmp_path / "uci_audit_metrics.csv", index=False)
    calibration.to_csv(tmp_path / "uci_audit_calibration.csv", index=False)
    (tmp_path / "uci_model.json").write_text(json.dumps(MODEL_INFO), encoding="utf-8")
    egfr_summary.to_csv(tmp_path / "egfr_summary.csv", index=False)

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
    spec = importlib.util.spec_from_file_location("build_report", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)  # no real .env, no site/ in the repo
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    module.main(["--outputs", str(tmp_path), "--llm", "--site"])
    assert "No ANTHROPIC_API_KEY found" in capsys.readouterr().out
    html = (tmp_path / "uci_report.html").read_text(encoding="utf-8")
    assert "Summary not generated" in html
    assert "Synthetic patients (Synthea)" in html  # eGFR section picked up from outputs/
    assert (tmp_path / "site" / "index.html").read_text(encoding="utf-8") == html
    facts = json.loads((tmp_path / "uci_facts.json").read_text(encoding="utf-8"))
    assert "age_band.<50.auc" in facts["facts"]  # "<50" stayed a string label


def test_summary_banner_says_ai_generated_and_methods_names_the_model(audited):
    metrics, calibration = audited
    summary = summarise(FACTS_JSON, FakeClient([GOOD]))
    html = build_report(metrics, calibration, summary=summary)
    assert "AI-generated summary" in html
    assert "Written by" not in html
    assert f"<dt>Summary model</dt><dd>{MODEL} via the Anthropic API" in html


def test_report_carries_the_disclaimers(audited):
    metrics, calibration = audited
    html = " ".join(build_report(metrics, calibration).split())
    assert "It is not medical advice, has not been clinically validated" in html
    assert "must not be used to make or influence decisions about individual patients" in html
    assert "its numbers describe that dataset, not any individual" in html
    assert "never publish it, including on GitHub Pages" in html
    assert "does not certify a model as fair, safe or compliant" in html
    assert 'provided "as is" under the MIT License, without warranty' in html
    assert "Not affiliated with or endorsed by" in html
    # A generic report must not claim its data was public or synthetic.
    assert "public or synthetic data only" not in html


# ---------------------------------------------------------------- small-cell safeguards

from equitypanel.report.facts import MIN_CELL, suppress_small_cells  # noqa: E402


def _cells_json():
    """Race: Asian has 8 readmitted (small); sex: Unknown has 2 patients (small)."""
    facts = {}
    for col, group, n, n_pos in [
        ("race", "White", 1000, 100),
        ("race", "Black", 400, 40),
        ("race", "Other", 60, 12),
        ("race", "Asian", 50, 8),
        ("sex", "Female", 760, 80),
        ("sex", "Male", 748, 79),
        ("sex", "Unknown", 2, 1),
    ]:
        facts[f"{col}.{group}.n"] = {"value": n, "n": n}
        facts[f"{col}.{group}.n_pos"] = {"value": n_pos, "n": n}
        facts[f"{col}.{group}.fnr"] = {"value": 0.5, "n": n, "ci_95": [0.4, 0.6]}
    facts["race.Asian.fnr_gap"] = {"value": 0.1, "n": 50}
    facts["model.n_test"] = {"value": 1510}
    facts["model.prevalence_test"] = {"value": 0.1}
    facts["model.auc_test"] = {"value": 0.64}
    facts["egfr.Black.waitlist_le20_newly_below"] = {"value": 7}
    facts["egfr.Black.waitlist_le20_2021"] = {"value": 89}
    facts["egfr.Black.n_higher"] = {"value": 0}
    return {"context": {"task": "x"}, "facts": facts}


def test_suppression_removes_small_groups_and_a_complementary_group():
    original = _cells_json()
    payload, suppressed = suppress_small_cells(original)
    # Asian (8 readmitted) and Unknown sex (2 patients) are small; each is the only small
    # group in its column, so the next-smallest group is suppressed too.
    assert suppressed[:4] == ["race.Asian", "race.Other", "sex.Unknown", "sex.Male"]
    facts = payload["facts"]
    for group in ("race.Asian", "race.Other", "sex.Unknown", "sex.Male"):
        assert not any(fid.startswith(group + ".") for fid in facts)
    assert "race.White.fnr" in facts and "sex.Female.fnr" in facts
    # Totals would let a hidden cell be recovered by subtraction.
    assert "model.n_test" not in facts and "model.prevalence_test" not in facts
    assert "model.auc_test" in facts
    # eGFR counts of 1-10 go; 0 and large counts stay.
    assert "egfr.Black.waitlist_le20_newly_below" not in facts
    assert "egfr.Black.waitlist_le20_2021" in facts and "egfr.Black.n_higher" in facts
    assert payload["context"]["small_cell_suppression"]["min_cell"] == MIN_CELL == 11
    assert "race.Asian.n" in original["facts"]  # input not modified


def test_suppression_counts_non_events_too():
    data = {
        "context": {},
        "facts": {
            "race.A.n": {"value": 100},
            "race.A.n_pos": {"value": 95},  # 5 non-events
            "race.B.n": {"value": 200},
            "race.B.n_pos": {"value": 20},
            "race.C.n": {"value": 300},
            "race.C.n_pos": {"value": 30},
        },
    }
    _, suppressed = suppress_small_cells(data)
    assert suppressed == ["race.A", "race.B"]


def test_no_suppression_when_all_groups_are_large():
    data = {
        "context": {},
        "facts": {
            "race.A.n": {"value": 100},
            "race.A.n_pos": {"value": 50},
            "model.n_test": {"value": 100},
        },
    }
    payload, suppressed = suppress_small_cells(data)
    assert suppressed == [] and "model.n_test" in payload["facts"]


def test_checker_rejects_a_suppressed_number():
    payload, _ = suppress_small_cells(_cells_json())
    # A summary can't restate the hidden group's size: the fact isn't in the payload.
    assert "unknown" in check_finding(
        "2 patients have unknown sex.", ["sex.Unknown.n"], payload["facts"]
    )


def test_facts_carry_no_dataset_name(facts_json):
    assert "dataset" not in facts_json["context"]


def test_summary_records_the_suppression_threshold():
    payload, _ = suppress_small_cells(FACTS_JSON)
    summary = summarise(payload, FakeClient([GOOD]))
    assert summary.min_cell == 11
    assert Summary.from_dict(summary.to_dict()).min_cell == 11


def test_report_warns_about_tiny_groups_and_shows_credits(audited):
    metrics, calibration = audited
    tiny = metrics.copy()
    mask = (tiny["group"] == "Asian") & (tiny["metric"] == "n_pos")
    tiny.loc[mask, "value"] = 3
    html = " ".join(
        build_report(
            tiny, calibration, data_credits=["Data: Example (CC BY 4.0)."], footer_note="Hosted."
        ).split()
    )
    assert "Very small groups:</strong> Race: Asian. Each has fewer than 11 patients" in html
    assert "<p>Data: Example (CC BY 4.0).</p>" in html and "<p>Hosted.</p>" in html


def test_methods_state_whether_suppression_was_on(audited):
    metrics, calibration = audited
    on = Summary(findings=[], attempts=1, min_cell=11)
    off = Summary(findings=[], attempts=1, min_cell=None)
    assert (
        "Small-cell suppression before sending: on (groups or counts of 1 to 10 removed)"
        in " ".join(build_report(metrics, calibration, summary=on).split())
    )
    assert "Small-cell suppression before sending: off (public data)" in " ".join(
        build_report(metrics, calibration, summary=off).split()
    )


def _load_script():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_report.py"
    spec = importlib.util.spec_from_file_location("build_report_llm", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_outputs(tmp_path, audited):
    metrics, calibration = audited
    metrics = metrics.copy()
    # Give Asian 8 readmitted patients so small-cell suppression has something to remove.
    metrics.loc[(metrics["group"] == "Asian") & (metrics["metric"] == "n_pos"), "value"] = 8
    metrics.to_csv(tmp_path / "uci_audit_metrics.csv", index=False)
    calibration.to_csv(tmp_path / "uci_audit_calibration.csv", index=False)
    (tmp_path / "uci_model.json").write_text(json.dumps(MODEL_INFO), encoding="utf-8")


def test_llm_flow_suppresses_shows_payload_and_asks(audited, tmp_path, monkeypatch, capsys):
    _write_outputs(tmp_path, audited)
    module = _load_script()
    client = FakeClient([{"text": "AUC is 0.71.", "fact_ids": ["model.auc_test"]}])
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "make_client", lambda: client)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")

    module.main(["--outputs", str(tmp_path), "--llm"])
    out = capsys.readouterr().out
    assert "aggregate facts, no patient rows" in out
    assert "Suppressed (fewer than 11 patients or events): race.Asian" in out
    payload = json.loads((tmp_path / "llm_payload.json").read_text(encoding="utf-8"))
    sent = json.loads(client.calls[0]["messages"][0]["content"].split("\n", 1)[1])
    assert sent == payload  # what was saved is exactly what was sent
    assert not any(fid.startswith("race.Asian.") for fid in sent["facts"])
    assert json.loads((tmp_path / "uci_summary.json").read_text(encoding="utf-8"))["min_cell"] == 11


def test_llm_flow_sends_nothing_without_confirmation(audited, tmp_path, monkeypatch, capsys):
    _write_outputs(tmp_path, audited)
    module = _load_script()
    client = FakeClient([GOOD])
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "make_client", lambda: client)
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # just Enter = no

    module.main(["--outputs", str(tmp_path), "--llm"])
    assert client.calls == []
    assert "Not sent" in capsys.readouterr().out
    assert "Summary not generated" in (tmp_path / "uci_report.html").read_text(encoding="utf-8")

"""End-to-end: split the cohort, fit the baseline, score the test set, audit it, write files."""

import json
from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd

from equitypanel.metrics.audit import audit
from equitypanel.metrics.core import auc
from equitypanel.model.baseline import (
    AUDIT_GROUP_COLS,
    _check_share,
    capacity_threshold,
    fit_baseline,
    model_weights,
    predict_risk,
    split_cohort,
    to_audit_frame,
)

OUTPUT_FILES = (
    "uci_scores.csv",
    "uci_audit_metrics.csv",
    "uci_audit_calibration.csv",
    "uci_model.json",
)
N_TOP_WEIGHTS = 15


def run_audit(
    cohort: pd.DataFrame,
    out_dir: str | PathLike,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    flag_share: float = 0.10,
) -> dict:
    """Train on 70% of patients, audit the model on the other 30%, write OUTPUT_FILES.

    The flagging threshold comes from the training scores, so the test set is only
    used once, for the audit. Nothing is written if any step fails.
    Returns the summary that is also saved as uci_model.json.
    """
    _check_share(flag_share, "flag_share")
    train, test = split_cohort(cohort, seed=seed)
    model = fit_baseline(train)
    threshold = capacity_threshold(predict_risk(model, train), flag_share)

    scores = predict_risk(model, test)
    frame = to_audit_frame(test, scores)
    report = audit(frame, list(AUDIT_GROUP_COLS), threshold, n_boot=n_boot, seed=seed)

    top = model_weights(model).head(N_TOP_WEIGHTS)
    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "prevalence_test": float(test["y_true"].mean()),
        "auc_test": auc(test["y_true"], scores),
        "threshold": threshold,
        "flag_share_test": float(np.mean(scores >= threshold)),
        "top_weights": [
            {"feature": str(f), "weight": float(w)}
            for f, w in zip(top["feature"], top["weight"], strict=True)
        ],
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.insert(0, "patient_id", test["patient_id"].to_numpy())
    frame.to_csv(out / "uci_scores.csv", index=False)
    report.metrics.to_csv(out / "uci_audit_metrics.csv", index=False)
    report.calibration.to_csv(out / "uci_audit_calibration.csv", index=False)
    (out / "uci_model.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

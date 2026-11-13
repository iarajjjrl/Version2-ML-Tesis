"""Regresión de disparador relativo: pruebas sintéticas, NO resultados IEEE-CIS."""
import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.v2.controller import MaintenanceController, PolicySettings
from src.experiments.v2.raw_windows import build_raw_windows
from src.experiments.v2.run_comparison_v2 import run_comparison


def test_absolute_mode_keeps_prior_behavior():
    c = MaintenanceController(PolicySettings("event", n_windows=8), "base")
    assert c.complete_window(1, .44).execute is False
    assert c.complete_window(2, .43).execute is True


def test_relative_mode_does_not_confuse_low_stable_f1_with_drift():
    c = MaintenanceController(
        PolicySettings("event", n_windows=12, trigger_mode="relative",
                       reference_f1=.45, relative_drop=.15),
        "base")
    for k, f1 in enumerate((.44, .43, .44, .42), 1):
        row = c.complete_window(k, f1)
        assert not row.execute
        assert row.reference_f1 == .45
        assert row.trigger_threshold == pytest.approx(.3825)
    assert c.complete_window(5, .35).execute is False
    next_row = c.complete_window(6, .34)
    assert next_row.execute
    assert next_row.reason == "event_due"


def test_relative_reference_uses_first_scored_window_after_promotion():
    c = MaintenanceController(
        PolicySettings("event", n_windows=12, trigger_mode="relative",
                       reference_f1=.45, relative_drop=.15),
        "base")
    c.complete_window(1, .44)
    c.complete_window(2, .35)
    train = c.complete_window(3, .34)
    assert train.execute
    c.register_candidate(3, "updated")
    promotion = c.complete_window(
        4, .34, candidate_f1=.55, incumbent_f1=.34)
    assert promotion.candidate_approved is True
    assert promotion.scoring_model == "base"
    assert promotion.model_for_next_window == "updated"
    assert promotion.reference_f1 == .45
    next_window = c.complete_window(5, .52)
    assert next_window.scoring_model == "updated"
    assert not next_window.execute
    assert next_window.reason == "reference_established_from_first_scoring_window"
    assert next_window.reference_f1 == pytest.approx(.52)
    assert next_window.trigger_threshold == pytest.approx(.442)
    assert c.complete_window(6, .40).execute is False
    assert c.complete_window(7, .42).execute is True


def test_rejected_candidate_does_not_change_reference():
    c = MaintenanceController(
        PolicySettings("event", n_windows=10, trigger_mode="relative",
                       reference_f1=.45, relative_drop=.15),
        "base")
    c.complete_window(1, .35)
    assert c.complete_window(2, .34).execute
    c.register_candidate(2, "loser")
    rejected = c.complete_window(3, .36, candidate_f1=.35, incumbent_f1=.36)
    assert rejected.candidate_approved is False
    assert rejected.reference_f1 == pytest.approx(.45)
    assert c.active_model == "base"


@pytest.mark.parametrize("kwargs", [
    {"reference_f1": None},
    {"reference_f1": 0},
    {"reference_f1": -0.2},
    {"reference_f1": 1.4},
])
def test_relative_mode_requires_valid_reference(kwargs):
    with pytest.raises(ValueError):
        PolicySettings("event", trigger_mode="relative", **kwargs)


def test_relative_runner_saves_reference_and_keeps_controls(tmp_path):
    n = 1200
    i = np.arange(n)
    raw = pd.DataFrame({
        "TransactionID": 100000 + i,
        "TransactionDT": 1 + i,
        "TransactionAmt": 10.0 + i % 137,
        "card1": 1000 + i % 11,
        "ProductCD": np.where(i % 2, "W", "C"),
        "C13": i % 9,
        "isFraud": (i % 5 == 0).astype(int),
    })
    base, holdout, windows, manifest = build_raw_windows(
        raw, max_rows=1000, n_windows=4, seed=42)
    inp = tmp_path / "input"
    inp.mkdir()
    base.to_csv(inp / "baseline_train_raw.csv", index=False)
    holdout.to_csv(inp / "diagnostic_holdout_raw.csv", index=False)
    for frame, info in zip(windows, manifest["windows"]):
        frame.to_csv(inp / info["file"], index=False)
    (inp / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    out = tmp_path / "new-results"
    summary = run_comparison(inp, out, trials=1, seed=42,
                             event_mode="relative", relative_drop=.15)
    assert set(summary.policy) == {"static", "periodic", "event_relative"}
    config = json.loads((out / "configuration.json").read_text(encoding="utf-8"))
    assert config["event_mode"] == "relative"
    assert config["initial_event_reference_f1"] == config[
        "diagnostic_holdout_metrics"]["f1"]
    rows = pd.read_csv(out / "comparison.csv")
    relative_rows = rows.loc[rows.policy == "event_relative"]
    assert len(relative_rows) == 4
    assert relative_rows.iloc[0].reference_f1 == pytest.approx(
        config["initial_event_reference_f1"])
    assert relative_rows.iloc[0].trigger_threshold == pytest.approx(
        config["initial_event_reference_f1"] * .85)

"""Pruebas v2 sin IEEE-CIS, con datos sintéticos. NO prueban calidad real."""
import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.v2.raw_windows import build_raw_windows
from src.experiments.v2.trainer import train_candidate, FAMILIES
from src.experiments.v2.run_comparison_v2 import run_comparison


def fake_ieee(n=1200):
    i = np.arange(n)
    # Ambas clases presentes en los bloques cronológicos sin muestreo aleatorio.
    return pd.DataFrame({
        "TransactionID": i + 100000,
        "TransactionDT": i + 1,
        "TransactionAmt": (i % 137).astype(float) + 10.0,
        "card1": 1000 + (i % 11),
        "ProductCD": np.where(i % 2, "W", "C"),
        "C13": i % 9,
        "isFraud": (i % 5 == 0).astype(int),
    })


def make_inputs(tmp_path):
    source = fake_ieee()
    base, holdout, windows, manifest = build_raw_windows(
        source, max_rows=1000, n_windows=4, seed=42)
    folder = tmp_path / "raw"
    folder.mkdir()
    base.to_csv(folder / "baseline_train_raw.csv", index=False)
    holdout.to_csv(folder / "diagnostic_holdout_raw.csv", index=False)
    for window, info in zip(windows, manifest["windows"]):
        window.to_csv(folder / info["file"], index=False)
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder, base, manifest


def test_trainer_uses_five_families_and_predicts_raw(tmp_path):
    _, baseline, manifest = make_inputs(tmp_path)
    bundle = train_candidate(baseline, features=manifest["features"],
                             trials=1, seed=42)
    assert set(row["family"] for row in bundle.search_results) == set(FAMILIES)
    assert bundle.family in FAMILIES
    assert len(bundle.predict(baseline.iloc[:4])) == 4
    assert "TransactionDT" not in bundle.features
    assert "TransactionID" not in bundle.features
    assert "Class" not in bundle.features
    assert 0.15 <= bundle.threshold <= 0.85
    result = bundle.evaluate(baseline.iloc[:50])
    assert result["tp"] + result["fp"] + result["tn"] + result["fn"] == 50


def test_runner_preserves_history_and_isolates_outputs(tmp_path):
    raw, _, _ = make_inputs(tmp_path)
    out = tmp_path / "results"
    summary = run_comparison(raw, out, seed=42, trials=1, every=2,
                             f1_trigger=1.0, consecutive_low=2)
    assert set(summary.policy) == {"static", "periodic", "event"}
    assert set((out / "comparison.csv").read_text().splitlines()[0].split(","))
    cfg = json.loads((out / "configuration.json").read_text(encoding="utf-8"))
    assert cfg["trials_per_family"] == 1
    assert cfg["initial_training_seconds_excluded"] >= 0
    assert (out / "candidate_artifacts" / "baseline.pkl").exists()
    window_rows = pd.read_csv(out / "comparison.csv")
    assert len(window_rows) == 12
    for policy in ("static", "periodic", "event"):
        sample = window_rows[window_rows.policy == policy].reset_index(drop=True)
        assert len(sample) == 4
        # Ninguna inferencia anterior puede utilizar el futuro.
        assert sample.loc[0, "scoring_model_id"] == "baseline"
        assert sample.loc[1, "scoring_model_id"] == "baseline"
        assert sample["training_seconds"].ge(0).all()
    # Nunca sobrescribe una ejecución existente.
    with pytest.raises(FileExistsError):
        run_comparison(raw, out, seed=42, trials=1)


def test_disjoint_epochs_are_chronological_and_do_not_mutate_source(tmp_path):
    from src.experiments.v2.prepare_epochs import partition_epochs, prepare_epochs
    source = fake_ieee(3000)
    before = source.copy(deep=True)
    epochs = partition_epochs(source, count=3)
    assert len(epochs) == 3
    assert sum(len(x) for x in epochs) == len(source)
    assert epochs[0].TransactionDT.max() < epochs[1].TransactionDT.min()
    assert epochs[1].TransactionDT.max() < epochs[2].TransactionDT.min()
    result = prepare_epochs(source, tmp_path / "epochs", epochs=3,
                            max_rows=1000, windows=4, seed=42)
    assert result["n_epochs"] == 3
    for epoch in range(1, 4):
        for scenario in ("natural", "synthetic"):
            assert (tmp_path / "epochs" / f"epoch_{epoch:02d}" / scenario / "manifest.json").exists()
    pd.testing.assert_frame_equal(source, before)

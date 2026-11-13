"""Pruebas sin IEEE-CIS ni red: partición sellada y evaluación fuera de muestra B."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.experiments.study_b_independent import (
    prepare_from_dataframe, evaluate_pair, _verify_sealed
)


def fake_ieee(n=1200):
    i = np.arange(n)
    return pd.DataFrame({
        "TransactionID": 1000000 + i,
        "TransactionDT": i + 1,
        "TransactionAmt": (i % 100 + 20).astype(float),
        "card1": 1000 + (i % 25),
        "ProductCD": np.where(i % 2, "C", "W"),
        "C13": i % 15,
        "isFraud": (i % 19 == 0).astype(int),
    })


def test_prepare_is_chronological_sealed_and_nonoverlapping(tmp_path):
    stage = tmp_path / "prepared"
    m = prepare_from_dataframe(fake_ieee(), stage, max_rows=1200, seed=42)
    assert m["selected_records"] == 1200
    assert [m["files"][f]["n"] for f in (
        "baseline_train.csv", "nominal.csv", "adaptation_shifted.csv",
        "future_natural.csv")] == [600, 120, 240, 240]
    assert m["files"]["adaptation_shifted.csv"]["end_key"] < (
        m["files"]["future_natural.csv"]["start_key"])
    assert (m["files"]["future_natural.csv"]["transaction_ids_sha256"] ==
            m["files"]["future_shifted.csv"]["transaction_ids_sha256"])
    natural = pd.read_csv(stage / "future_natural.csv")
    shifted = pd.read_csv(stage / "future_shifted.csv")
    assert natural["Class"].equals(shifted["Class"])
    assert not natural["TransactionAmt"].equals(shifted["TransactionAmt"])
    assert _verify_sealed(stage) == m
    with pytest.raises(FileExistsError):
        prepare_from_dataframe(fake_ieee(), stage)


def test_tampering_future_is_detected(tmp_path):
    stage = tmp_path / "prepared"
    prepare_from_dataframe(fake_ieee(), stage)
    with (stage / "future_natural.csv").open("a", encoding="utf-8") as stream:
        stream.write("TAMPER")
    with pytest.raises(ValueError, match="alterado"):
        _verify_sealed(stage)


def test_evaluation_compares_same_future_without_dispatch(tmp_path):
    stage = tmp_path / "prepared"
    prepare_from_dataframe(fake_ieee(), stage)
    natural = pd.read_csv(stage / "future_natural.csv")
    X = natural.drop(columns="Class")
    y = natural["Class"]
    before = DummyClassifier(strategy="constant", constant=0).fit(X, y)
    after = DummyClassifier(strategy="constant", constant=1).fit(X, y)
    old_path = tmp_path / "old.pkl"
    new_path = tmp_path / "new.pkl"
    joblib.dump(before, old_path)
    joblib.dump(after, new_path)
    report = evaluate_pair(stage, old_path, new_path,
                           output=tmp_path / "result.json")
    assert set(report["future_results"]) == {
        "future_natural.csv", "future_shifted.csv"}
    for entry in report["future_results"].values():
        assert entry["before"]["tp"] == 0
        assert entry["before"]["fn"] == entry["positives"]
        assert entry["after"]["tp"] == entry["positives"]
        assert entry["after"]["fn"] == 0
    assert json.loads((tmp_path / "result.json").read_text(
        encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        evaluate_pair(stage, old_path, new_path,
                      output=tmp_path / "result.json")
    with pytest.raises(ValueError, match="misma huella"):
        evaluate_pair(stage, old_path, old_path,
                      output=tmp_path / "another.json")


def test_reject_unsorted_raw_data(tmp_path):
    with pytest.raises(ValueError, match="cronológico"):
        prepare_from_dataframe(fake_ieee().iloc[::-1],
                               tmp_path / "bad")

"""Unit tests: NO require IEEE-CIS files or run Optuna/retraining."""
import pandas as pd
import pytest

from src.experiments.v2.controller import MaintenanceController, PolicySettings
from src.experiments.v2.temporal import split_temporally, fit_transform_blocks
from src.prepare_data import FraudFeaturePreprocessor


def frame(n=40):
    return pd.DataFrame({
        "TransactionDT": range(1, n + 1),
        "TransactionID": range(1000, 1000 + n),
        "TransactionAmt": [float(i + 10) for i in range(n)],
        "card1": [1001 if i % 2 == 0 else 1002 for i in range(n)],
        "ProductCD": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "Class": [i % 2 for i in range(n)],
    })


def test_temporal_splits_are_disjoint_and_chronological():
    b = split_temporally(frame())
    assert [len(b.train), len(b.tune), len(b.calibrate)] == [28, 6, 6]
    assert b.train.TransactionDT.max() < b.tune.TransactionDT.min()
    assert b.tune.TransactionDT.max() < b.calibrate.TransactionDT.min()
    assert not (set(b.train.TransactionID) & set(b.calibrate.TransactionID))


def test_unsorted_or_duplicate_data_rejected():
    raw = frame()
    with pytest.raises(ValueError, match="ordenado"):
        split_temporally(raw.iloc[::-1])
    bad = raw.copy()
    bad.loc[2, "TransactionID"] = 1001
    with pytest.raises(ValueError):
        split_temporally(bad)


def test_preprocessor_does_not_fit_future_data():
    b = split_temporally(frame())
    b.tune.loc[:, "TransactionAmt"] = 100000
    b.calibrate.loc[:, "TransactionAmt"] = 999999
    changed = fit_transform_blocks(b, FraudFeaturePreprocessor,
                                   ["TransactionAmt", "card1", "ProductCD"])
    assert changed.preprocessor.card1_amount_means_[1001] < 100
    assert changed.tune_x.columns.tolist() == changed.train_x.columns.tolist()
    # Modificar el futuro no altera el estado aprendido de entrenamiento.
    assert changed.preprocessor.card1_counts_[1001] == 14
    assert changed.calibrate_x["TransactionAmt"].min() == 999999


def test_forbidden_target_and_time_features():
    b = split_temporally(frame())
    with pytest.raises(ValueError, match="etiqueta"):
        fit_transform_blocks(b, FraudFeaturePreprocessor, ["Class"])
    with pytest.raises(ValueError, match="identificador"):
        fit_transform_blocks(b, FraudFeaturePreprocessor, ["TransactionID"])


@pytest.mark.parametrize("every,expected", [(2, [2, 4, 6]), (3, [3, 6]), (4, [4])])
def test_periodic_schedules_and_promotion_timing(every, expected):
    c = MaintenanceController(PolicySettings("periodic", every=every), "base")
    trained = []
    for k in range(1, 9):
        if c.pending is not None:
            row = c.complete_window(k, .5, candidate_f1=.7, incumbent_f1=.5)
            assert row.scoring_model != f"candidate_{trained[-1]}"
            assert row.candidate_approved is True
        else:
            row = c.complete_window(k, .5)
        if row.execute:
            trained.append(k)
            c.register_candidate(k, f"candidate_{k}")
        if k in expected:
            assert row.execute
    assert trained == expected
    assert c.promotions == len(expected)


def test_model_promoted_at_k_plus_one_used_only_from_k_plus_two():
    c = MaintenanceController(PolicySettings("periodic", every=2), "initial")
    r1 = c.complete_window(1, .5)
    r2 = c.complete_window(2, .5)
    assert not r1.execute and r2.execute
    c.register_candidate(2, "candidate")
    r3 = c.complete_window(3, .2, candidate_f1=.7, incumbent_f1=.5)
    assert r3.scoring_model == "initial"  # Score k+1 before promotion
    assert r3.model_for_next_window == "candidate"
    r4 = c.complete_window(4, .3)
    assert r4.scoring_model == "candidate"  # Only k+2 uses new model


def test_candidate_rejected_below_margin_and_exact_tie():
    c = MaintenanceController(PolicySettings("periodic", every=2), "base")
    c.complete_window(1, .5)
    c.complete_window(2, .5)
    c.register_candidate(2, "bad")
    r = c.complete_window(3, .4, candidate_f1=.509, incumbent_f1=.5)
    assert r.candidate_reviewed and r.candidate_approved is False
    assert c.active_model == "base"


def test_event_threshold_is_strict_and_requires_persistence():
    c = MaintenanceController(PolicySettings("event", consecutive_low=2), "base")
    assert not c.complete_window(1, .50).execute
    assert not c.complete_window(2, .49).execute
    r3 = c.complete_window(3, .48)
    assert r3.execute
    c.record_training_failure(3)
    assert not c.complete_window(4, .60).execute


def test_end_horizon_prevents_unmeasurable_promotion():
    c = MaintenanceController(PolicySettings("periodic", every=1), "base")
    for k in range(1, 9):
        if c.pending:
            result = c.complete_window(k, .5, candidate_f1=.7, incumbent_f1=.5)
        else:
            result = c.complete_window(k, .5)
        if result.execute:
            c.register_candidate(k, f"c{k}")
    assert c.last_attempt <= 6
    assert c.attempts == 3  # k=1,3,5: k=2,4,6 blocked by review


def test_budget_cap_and_missing_promotion_scores():
    c = MaintenanceController(PolicySettings("periodic", every=1, max_attempts=1), "base")
    r = c.complete_window(1, .2)
    assert r.execute
    c.register_candidate(1, "candidate")
    with pytest.raises(ValueError, match="métricas"):
        c.complete_window(2, .3)
    # Validación fallida no consume ventana 2.
    good = c.complete_window(2, .3, candidate_f1=.7, incumbent_f1=.4)
    assert good.candidate_approved
    assert not c.complete_window(3, .2).execute


def test_concurrent_registration_is_prohibited():
    c = MaintenanceController(PolicySettings("periodic", every=1), "base")
    assert c.complete_window(1, .5).execute
    with pytest.raises(RuntimeError):
        c.complete_window(2, .5)
    c.record_training_failure(1)
    next_step = c.complete_window(2, .5)
    assert not next_step.execute and next_step.reason == "cooldown"


def test_raw_windows_do_not_preprocess_and_preserve_time():
    from src.experiments.v2.raw_windows import build_raw_windows
    raw = frame(1200).rename(columns={"Class": "isFraud"})
    b, holdout, wins, m = build_raw_windows(raw, max_rows=1000, n_windows=4,
                                            inject_drift_from=3, seed=42)
    assert len(b) == 500 and len(holdout) == 100
    assert sum(map(len, wins)) == 400
    assert m["preprocessed"] is False
    assert "TransactionDT" in b and "Class" in b and "card1_count" not in b
    assert max(b.TransactionDT) < min(holdout.TransactionDT)
    assert max(holdout.TransactionDT) < min(wins[0].TransactionDT)
    assert not m["windows"][1]["synthetic_shift"]
    assert m["windows"][2]["synthetic_shift"]
    assert wins[0]["TransactionAmt"].max() < wins[2]["TransactionAmt"].min()


def test_raw_windows_original_is_not_mutated():
    from src.experiments.v2.raw_windows import build_raw_windows
    raw = frame(1200).rename(columns={"Class": "isFraud"})
    before = raw.copy(deep=True)
    build_raw_windows(raw, max_rows=1200, n_windows=4, seed=1, inject_drift_from=3)
    pd.testing.assert_frame_equal(raw, before)


def test_promoting_event_candidate_resets_old_model_low_streak():
    c = MaintenanceController(PolicySettings("event", consecutive_low=2), "base")
    c.complete_window(1, .3)
    assert c.complete_window(2, .3).execute
    c.register_candidate(2, "updated")
    reviewed = c.complete_window(3, .3, candidate_f1=.8, incumbent_f1=.3)
    assert reviewed.candidate_approved and c.low_streak == 0
    assert reviewed.reason == "new_model_not_evaluated_until_next_window"
    assert not c.complete_window(4, .3).execute
    assert c.complete_window(5, .3).execute

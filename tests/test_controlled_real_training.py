"""Prueba corta de entrenamiento REAL de cinco familias sobre datos artificiales.

No usa IEEE-CIS, no contiene transacciones reales y no valida superioridad
predictiva. Comprueba que el adaptador y el entrenador v2 pueden operar juntos.
"""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.mlops.controlled_v2 import bootstrap, step
from src.mlops.controller_bridge import load_state


class RealTrainerSmoke(unittest.TestCase):
    def test_training_and_four_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            store = root / "state"
            rng = np.random.default_rng(20261113)
            n_base, n_holdout, n_win, width = 320, 80, 4, 100
            n = n_base + n_holdout + n_win * width
            data = pd.DataFrame({
                "TransactionDT": np.arange(n) + 1,
                "TransactionID": np.arange(n) + 10001,
                "TransactionAmt": rng.lognormal(3.0, 1.0, n),
                "card1": rng.integers(100, 110, n),
                "C13": rng.poisson(3, n),
                "Class": rng.binomial(1, .22, n),
            })
            features = ["TransactionAmt", "card1", "C13"]
            data.iloc[:n_base].to_csv(raw / "baseline_train_raw.csv", index=False)
            data.iloc[n_base:n_base + n_holdout].to_csv(
                raw / "diagnostic_holdout_raw.csv", index=False)
            entries = []
            for k in range(1, n_win + 1):
                first = n_base + n_holdout + (k - 1) * width
                file = f"window_{k:02d}_raw.csv"
                part = data.iloc[first:first + width]
                part.to_csv(raw / file, index=False)
                entries.append({"window": k, "file": file, "records": width})
            (raw / "manifest.json").write_text(json.dumps({
                "dataset": "artificial_for_ci_only",
                "preprocessed": False,
                "features": features,
                "windows": entries,
            }), encoding="utf-8")
            init = bootstrap(raw, store, seed=42, trials=1,
                             f1_trigger=1.0, consecutive_low=2,
                             cooldown_windows=1, min_delta=.01)
            self.assertTrue((store / "models" / "baseline.pkl").exists())
            self.assertGreaterEqual(init["initial_seconds"], 0.0)
            observed = [step(store) for _ in range(n_win)]
            state, ctrl = load_state(store / "state.json")
            self.assertEqual(ctrl.last_window, n_win)
            self.assertEqual(len(observed), n_win)
            self.assertEqual(len(state["processed"]), n_win)
            self.assertGreaterEqual(ctrl.attempts, 1)
            self.assertTrue(any(event["train_attempt"] for event in observed))
            self.assertTrue(all(event["training_error"] is None for event in observed),
                            msg=[event["training_error"] for event in observed])


if __name__ == "__main__":
    unittest.main()

"""Prueba del adaptador completo con lotes ficticios, sin entrenar modelos.

No modifica el modelo vigente del repositorio. Verifica que las ventanas se
consumen una vez, que k+1 valida y que el cambio es visible desde k+2.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.mlops.controlled_v2 import bootstrap, step
from src.mlops.controller_bridge import load_state


class FakeBundle:
    family = "RandomForest"
    def __init__(self, candidate=False):
        self.candidate = candidate
        self.threshold = .50
        self.seed = 42
        self.tuning_f1 = .60
        self.calibration_f1 = .60

    def evaluate(self, frame):
        first = int(frame["TransactionID"].iloc[0])
        if first < 100:
            f1 = .80
        elif self.candidate:
            f1 = .75
        else:
            f1 = .40
        return {"tp": 2, "fp": 1, "fn": 1, "tn": len(frame) - 4,
                "f1": f1, "precision": .66, "recall": .66}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.raw = Path(self.root.name) / "raw"
        self.raw.mkdir()
        self.store = Path(self.root.name) / "store"
        manifest = {
            "dataset": "demo sintética",
            "preprocessed": False,
            "features": ["TransactionAmt", "card1", "C13"],
            "windows": [],
        }
        def frame(ids):
            return pd.DataFrame({
                "TransactionDT": ids,
                "TransactionID": ids,
                "TransactionAmt": [float(v) for v in ids],
                "card1": [101] * len(ids),
                "C13": [3] * len(ids),
                "Class": [int(v % 2 == 0) for v in ids],
            })
        frame(range(1, 21)).to_csv(self.raw / "baseline_train_raw.csv", index=False)
        frame(range(21, 31)).to_csv(self.raw / "diagnostic_holdout_raw.csv", index=False)
        for k in range(1, 9):
            ids = range(100 + 10 * k, 105 + 10 * k)
            fname = f"window_{k:02d}_raw.csv"
            frame(ids).to_csv(self.raw / fname, index=False)
            manifest["windows"].append({"window": k, "file": fname, "records": 5})
        (self.raw / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")

    def test_full_state_and_candidate_lifecycle(self):
        trained = []
        def fake_train(history, *, features, trials, seed, adaptation_window=None):
            trained.append((len(history), None if adaptation_window is None else len(adaptation_window)))
            return FakeBundle(candidate=adaptation_window is not None)
        with patch("src.mlops.controlled_v2.train_candidate", side_effect=fake_train):
            initial = bootstrap(self.raw, self.store, trials=1)
            self.assertEqual(initial["active_model_id"], "baseline")
            first = step(self.store)
            self.assertEqual(first["window"], 1)
            self.assertFalse(first["train_attempt"])
            second = step(self.store)
            self.assertEqual(second["window"], 2)
            self.assertTrue(second["train_attempt"])
            self.assertEqual(second["candidate_created"], "event_candidate_after_02")
            self.assertEqual(trained, [(20, None), (35, 5)])
            third = step(self.store)
            self.assertTrue(third["candidate_approved"])
            self.assertEqual(third["scoring_model"], "baseline")
            self.assertEqual(third["active_after_window"], "event_candidate_after_02")
            fourth = step(self.store)
            self.assertEqual(fourth["scoring_model"], "event_candidate_after_02")
            data, ctrl = load_state(self.store / "state.json")
            self.assertEqual(ctrl.promotions, 1)
            self.assertEqual(len(data["processed"]), 4)
            self.assertEqual(data["processed"][2]["candidate_f1"], .75)

    def test_changed_manifest_is_rejected(self):
        with patch("src.mlops.controlled_v2.train_candidate",
                   return_value=FakeBundle()):
            bootstrap(self.raw, self.store, trials=1)
            manifest_path = self.raw / "manifest.json"
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifiesto"):
                step(self.store)


if __name__ == "__main__":
    unittest.main()

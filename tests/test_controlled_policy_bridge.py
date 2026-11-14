"""Pruebas sin dataset real del controlador COMPARTIDO con el Estudio A."""
import unittest

from src.experiments.v2.controller import PolicySettings
from src.mlops.controller_bridge import (
    complete_observation, controller_from_dict, create_state,
)


class ControlledPolicyTests(unittest.TestCase):
    def make(self, n=8):
        settings = PolicySettings(
            policy="event", n_windows=n, f1_trigger=.50,
            consecutive_low=2, cooldown_windows=1, min_delta=.01)
        state = create_state(settings, "baseline")
        controller = controller_from_dict(settings, state["controller"])
        return state, controller

    def observed(self, state, controller, number, score, candidate=None, challenger=None,
                 trainer=None):
        return complete_observation(
            state, controller, window=number, observed_f1=score,
            candidate_f1=challenger, incumbent_f1=(score if challenger is not None else None),
            train_callback=trainer)

    def test_request_k_compare_k_plus_1_use_k_plus_2(self):
        state, controller = self.make()
        self.observed(state, controller, 1, .7)
        self.observed(state, controller, 2, .4)
        third = self.observed(state, controller, 3, .4, trainer=lambda w: "candidate")
        self.assertTrue(third["train_attempt"])
        self.assertEqual(controller.pending, (3, "candidate"))
        restarted = controller_from_dict(PolicySettings(**state["settings"]),
                                         state["controller"])
        fourth = self.observed(state, restarted, 4, .40, challenger=.52)
        self.assertTrue(fourth["candidate_approved"])
        self.assertEqual(fourth["scoring_model"], "baseline")
        self.assertEqual(fourth["active_after_window"], "candidate")
        fifth = self.observed(state, restarted, 5, .60)
        self.assertEqual(fifth["scoring_model"], "candidate")
        self.assertEqual(restarted.promotions, 1)

    def test_rejected_candidate_does_not_replace_baseline(self):
        state, c = self.make()
        self.observed(state, c, 1, .30)
        self.observed(state, c, 2, .30, trainer=lambda _: "candidate")
        third = self.observed(state, c, 3, .40, challenger=.405)
        self.assertFalse(third["candidate_approved"])
        self.assertEqual(c.active_model, "baseline")

    def test_missing_training_is_recorded_as_failure(self):
        state, c = self.make()
        self.observed(state, c, 1, .30)
        result = self.observed(state, c, 2, .30, trainer=lambda _: 1 / 0)
        self.assertEqual(c.attempts, 1)
        self.assertIsNone(c.pending)
        self.assertIn("ZeroDivisionError", result["training_error"])

    def test_replay_is_rejected(self):
        state, c = self.make()
        self.observed(state, c, 1, .75)
        with self.assertRaises(ValueError):
            self.observed(state, c, 1, .75)

    def test_no_late_training_without_two_future_windows(self):
        state, c = self.make(n=3)
        self.observed(state, c, 1, .40)
        result = self.observed(state, c, 2, .40, trainer=lambda _: "late")
        self.assertTrue(result["requested"])
        self.assertFalse(result["train_attempt"])
        self.assertEqual(c.attempts, 0)


if __name__ == "__main__":
    unittest.main()

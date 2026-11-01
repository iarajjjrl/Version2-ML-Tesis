"""Pruebas rápidas para las reglas de decisión del experimento.

Ejecutar: python -m unittest discover -s tests -p "test_experiment_policies.py"
"""
import unittest
from src.experiments.policies import approve_promotion, decide_update


class PolicyTests(unittest.TestCase):
    def test_static_does_not_retrain(self):
        actual = decide_update("static", 4, .01, .50, 4, 1, 0, 0, None)
        self.assertFalse(actual[0])

    def test_periodic_only_at_fixed_boundaries(self):
        for i in range(1, 9):
            should = decide_update("periodic", i, .91, .50, 4, 1, 0, 0, None)[0]
            self.assertEqual(should, i in (4, 8))

    def test_event_requires_consecutive_lows(self):
        first = decide_update("event", 1, .45, .50, 4, 2, 0, 0, None)
        self.assertEqual(first[:2], (False, 1))
        second = decide_update("event", 2, .42, .50, 4, 2, 0, first[1], None)
        self.assertEqual(second[:2], (True, 0))

    def test_streak_resets_after_recovery(self):
        first = decide_update("event", 1, .45, .50, 4, 2, 0, 0, None)
        second = decide_update("event", 2, .61, .50, 4, 2, 0, first[1], None)
        self.assertEqual(second[:2], (False, 0))

    def test_cooldown_blocks_next_window(self):
        blocked = decide_update("event", 3, .30, .50, 4, 1, 1, 0, 2)
        self.assertEqual(blocked[0], False)
        self.assertEqual(blocked[2], "cooldown")
        allowed = decide_update("event", 4, .30, .50, 4, 1, 1, blocked[1], 2)
        self.assertTrue(allowed[0])

    def test_promotion_rejects_tie_or_worse(self):
        self.assertFalse(approve_promotion(.50, .50, 0, 0))
        self.assertFalse(approve_promotion(.49, .50, .01, 0))
        self.assertFalse(approve_promotion(.509, .50, .01, 0))
        self.assertTrue(approve_promotion(.52, .50, .01, 0))

    def test_promotion_enforces_minimum_f1(self):
        self.assertFalse(approve_promotion(.42, .40, .01, .50))


if __name__ == "__main__":
    unittest.main()

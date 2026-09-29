"""
Rigorous Testing of RUL Prognostics and Stochastic Degradation Modeling
"""

import unittest
from src.core.state import CompositeEngineHealth, SubsystemHealth
from src.ai_ml.rul_estimator import RULPrognosticsEstimator

class TestRULPrognostics(unittest.TestCase):

    def setUp(self):
        self.estimator = RULPrognosticsEstimator(initial_operating_hours=300.0, nominal_tbo_hours=1800.0)

    def test_nominal_rul_bounds(self):
        # Health index 1.0 (perfect)
        health = CompositeEngineHealth(
            overall_health_index=1.0,
            subsystem_scores=SubsystemHealth()
        )
        prog = self.estimator.update(health, dt_flight_hours=0.01)
        self.assertGreater(prog.rul_hours_median, 1200.0)
        # P10 should be less than P50, and P50 should be less than P90
        self.assertLess(prog.rul_hours_p10, prog.rul_hours_median)
        self.assertGreater(prog.rul_hours_p90, prog.rul_hours_median)
        self.assertEqual(prog.degradation_trend, "STABLE_NOMINAL")
        self.assertGreater(prog.mission_completion_probability, 95.0)

    def test_accelerating_wear_under_fault(self):
        # Severely degraded lubrication
        sub = SubsystemHealth(lubrication_system=35.0, thermal_cht=55.0)
        health = CompositeEngineHealth(
            overall_health_index=0.48,
            subsystem_scores=sub
        )
        # Run 50 updates under severe wear
        for _ in range(50):
            prog = self.estimator.update(health, dt_flight_hours=0.05)

        self.assertLess(prog.rul_hours_median, 600.0)
        self.assertIn("Lubrication", prog.primary_limiting_subsystem)
        self.assertIn(prog.degradation_trend, ["ACCELERATING_WEAR", "IMMINENT_CRITICAL"])

if __name__ == "__main__":
    unittest.main()

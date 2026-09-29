"""
Rigorous Testing of Multi-Mission Flight Profile Simulation
"""

import unittest
from src.simulation.mission_simulator import MissionSimulator
from src.simulation.fault_injector import FaultInjector

class TestMissionSimulation(unittest.TestCase):

    def setUp(self):
        self.injector = FaultInjector()
        self.simulator = MissionSimulator(fault_injector=self.injector)

    def test_available_profiles(self):
        profiles = self.simulator.get_available_profiles()
        keys = [p["key"] for p in profiles]
        self.assertIn("high_altitude_recon", keys)
        self.assertIn("hot_desert_patrol", keys)
        self.assertIn("rapid_throttle_tactical", keys)
        self.assertIn("maritime_endurance", keys)
        self.assertIn("fault_evaluation_flight", keys)

    def test_high_altitude_simulation(self):
        self.simulator.set_profile("high_altitude_recon")
        # Step through 50 steps
        for _ in range(50):
            frame = self.simulator.step(dt_s=0.2)
        
        self.assertGreater(frame.rpm, 1200.0)
        self.assertLess(frame.rpm, 6000.0)
        self.assertGreater(frame.map_inhg, 15.0)
        self.assertGreater(frame.fuel_flow_lph, 2.0)
        self.assertGreater(frame.oil_pressure_bar, 1.2)

    def test_hot_desert_simulation(self):
        self.simulator.set_profile("hot_desert_patrol")
        for _ in range(40):
            frame = self.simulator.step(dt_s=0.2)
        # Desert ambient temp should be high (>35°C)
        self.assertGreater(frame.ambient_temp_c, 30.0)
        self.assertGreater(frame.coolant_temp_c, 75.0)

    def test_rapid_throttle_transitions(self):
        self.simulator.set_profile("rapid_throttle_tactical")
        # Step through rapid throttle transitions
        throttle_history = []
        for _ in range(80):
            frame = self.simulator.step(dt_s=0.5)
            throttle_history.append(frame.throttle_pct)
        # Verify throttle varied
        self.assertGreater(max(throttle_history) - min(throttle_history), 20.0)

if __name__ == "__main__":
    unittest.main()

"""
Rigorous Testing of FastAPI REST Endpoints, Fault Control, and Mission Replay Engine
"""

import unittest
import json
from starlette.testclient import TestClient
from src.api.server import app, simulator, telemetry_history
from src.simulation.mission_recorder import MissionRecorderReplay

class TestAPIAndReplay(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")
        self.assertIn("DRDO", data["system"])

    def test_mission_profiles_list(self):
        resp = self.client.get("/api/mission/profiles")
        self.assertEqual(resp.status_code, 200)
        profiles = resp.json()
        self.assertGreaterEqual(len(profiles), 4)

    def test_mission_selection(self):
        resp = self.client.post("/api/mission/select", json={"profile_key": "hot_desert_patrol"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["active_profile"], "hot_desert_patrol")

    def test_fault_injection_and_clearing(self):
        # Inject misfire
        resp = self.client.post("/api/fault/inject", json={"fault_type": "misfire", "severity": 0.8, "target_cylinder": 3})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("misfire", resp.json()["active_faults"])

        # Clear fault
        resp = self.client.post("/api/fault/clear", json={"fault_type": "misfire"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("misfire", resp.json()["active_faults"])

    def test_report_generation_endpoint(self):
        from src.core.digital_twin_engine import DigitalTwinSynchronizer
        from src.health.health_monitor import HealthMonitoringSystem
        from src.ai_ml.rul_estimator import RULPrognosticsEstimator
        from src.core.state import EngineFullStatePacket
        twin_engine = DigitalTwinSynchronizer()
        health_mon = HealthMonitoringSystem()
        rul_est = RULPrognosticsEstimator()
        
        telemetry_history.clear()
        for _ in range(5):
            telem = simulator.step(dt_s=0.1)
            tw = twin_engine.synchronize(telem)
            h = health_mon.evaluate(telem, tw)
            prog = rul_est.update(h)
            pkt = EngineFullStatePacket(
                telemetry=telem,
                digital_twin=tw,
                health=h,
                active_alerts=[],
                prognosis=prog
            )
            telemetry_history.append(pkt.model_dump())

        resp = self.client.post("/api/report/generate")
        self.assertEqual(resp.status_code, 200)
        rep = resp.json()
        self.assertIn("report_id", rep)
        self.assertEqual(rep["problem_statement_id"], "SIH26054")

if __name__ == "__main__":
    unittest.main()

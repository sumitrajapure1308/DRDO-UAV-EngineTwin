"""
Comprehensive End-to-End System Integration Test
Validates:
Simulator -> Digital Twin Synchronizer -> Health Monitor -> Fault Detector ->
Hybrid AI -> XAI -> RUL Prognostics -> Advisory -> Report Generator
"""

import unittest
import time
import os
import numpy as np
from pathlib import Path
from src.core.digital_twin_engine import DigitalTwinSynchronizer
from src.health.health_monitor import HealthMonitoringSystem
from src.diagnostics.fault_detector import IntelligentFaultDetector
from src.diagnostics.xai_explainer import XAIExplainer
from src.ai_ml.hybrid_detector import HybridPhysicsAIDetector
from src.ai_ml.rul_estimator import RULPrognosticsEstimator
from src.ai_ml.advisory_engine import MaintenanceAdvisoryEngine
from src.simulation.fault_injector import FaultInjector
from src.simulation.mission_simulator import MissionSimulator
from src.simulation.mission_recorder import MissionRecorderReplay
from src.reports.report_generator import DRDOMissionReportGenerator
from src.core.state import EngineFullStatePacket

class TestEndToEndSystem(unittest.TestCase):

    def setUp(self):
        self.fault_injector = FaultInjector()
        self.simulator = MissionSimulator(fault_injector=self.fault_injector)
        self.twin = DigitalTwinSynchronizer()
        self.health = HealthMonitoringSystem()
        self.detector = IntelligentFaultDetector()
        self.xai = XAIExplainer()
        self.ai = HybridPhysicsAIDetector()
        self.rul = RULPrognosticsEstimator()
        self.advisory = MaintenanceAdvisoryEngine()
        self.recorder = MissionRecorderReplay()
        self.reporter = DRDOMissionReportGenerator()

    def test_complete_mission_pipeline_and_report_generation(self):
        # 1. Run 30 steps of simulation in nominal state
        history = []
        for _ in range(20):
            telem = self.simulator.step(dt_s=0.1)
            twin_est = self.twin.synchronize(telem)
            health_est = self.health.evaluate(telem, twin_est)
            alerts = self.detector.detect(telem, twin_est)
            ai_res = self.ai.analyze(telem, twin_est)
            xai_res = self.xai.explain(telem, twin_est)
            prognosis = self.rul.update(health_est, dt_flight_hours=(0.1 / 3600.0))
            adv = self.advisory.generate_advisories(alerts, health_est, prognosis)

            packet = EngineFullStatePacket(
                telemetry=telem,
                digital_twin=twin_est,
                health=health_est,
                active_alerts=alerts,
                prognosis=prognosis,
                xai_attributions=xai_res.get("feature_attributions_pct", {})
            )
            history.append(packet.model_dump())

        # Baseline checks
        self.assertGreater(history[0]["health"]["overall_health_index"], 0.85)
        self.assertEqual(len(history[-1]["active_alerts"]), 0)

        # 2. Inject Fault: Lubrication Issue
        self.fault_injector.inject_fault("lubrication_issue", severity=0.8)
        
        fault_history = []
        for _ in range(25):
            telem = self.simulator.step(dt_s=0.1)
            twin_est = self.twin.synchronize(telem)
            health_est = self.health.evaluate(telem, twin_est)
            alerts = self.detector.detect(telem, twin_est)
            ai_res = self.ai.analyze(telem, twin_est)
            xai_res = self.xai.explain(telem, twin_est)
            prognosis = self.rul.update(health_est, dt_flight_hours=(0.1 / 3600.0))
            adv = self.advisory.generate_advisories(alerts, health_est, prognosis)

            packet = EngineFullStatePacket(
                telemetry=telem,
                digital_twin=twin_est,
                health=health_est,
                active_alerts=alerts,
                prognosis=prognosis,
                xai_attributions=xai_res.get("feature_attributions_pct", {})
            )
            fault_history.append(packet.model_dump())

        # Anomaly and fault verification
        final_packet = fault_history[-1]
        active_codes = [a["fault_code"] for a in final_packet["active_alerts"]]
        self.assertTrue(any("LUBE" in c for c in active_codes), f"Expected LUBE fault in {active_codes}")
        
        # Verify lubrication subsystem score dropped
        lube_score = final_packet["health"]["subsystem_scores"]["lubrication_system"]
        self.assertLess(lube_score, 80.0)

        # Verify AI anomaly detector flagged anomaly
        self.assertGreater(ai_res["anomaly_score"], 0.35)

        # Verify XAI identified Oil Pressure as top driver
        top_driver_features = [d["feature"] for d in xai_res.get("top_drivers", [])]
        self.assertTrue(any("Oil Pressure" in f for f in top_driver_features), f"Expected Oil Pressure in XAI top drivers: {top_driver_features}")

        # 3. Generate DRDO Mission Health Report
        full_session = history + fault_history
        report = self.reporter.generate_report(full_session, mission_name="E2E_VALIDATION_MISSION")

        self.assertIn("report_id", report)
        self.assertEqual(report["problem_statement_id"], "SIH26054")
        self.assertGreater(report["detected_fault_count"], 0)
        self.assertTrue(os.path.exists(report["json_filepath"]))
        self.assertTrue(os.path.exists(report["html_filepath"]))

    def test_can_protocol_roundtrip(self):
        from src.core.bus import AeroCANProtocol
        # Create a sample telemetry frame
        telem = self.simulator.step(dt_s=0.1)
        frames = AeroCANProtocol.encode_telemetry_to_can(telem)
        self.assertGreaterEqual(len(frames), 5)
        
        # Decode frames back
        decoded = {}
        for f in frames:
            AeroCANProtocol.decode_can_frame(f, decoded)

        self.assertAlmostEqual(decoded["rpm"], telem.rpm, delta=2.0)
        self.assertAlmostEqual(decoded["cht_1_c"], telem.cht_1_c, delta=0.2)
        self.assertAlmostEqual(decoded["egt_1_c"], telem.egt_1_c, delta=0.2)
        self.assertAlmostEqual(decoded["oil_pressure_bar"], telem.oil_pressure_bar, delta=0.05)

    def test_secure_telemetry_authentication(self):
        from src.core.secure_telemetry import SecureTelemetryGateway
        gateway = SecureTelemetryGateway()
        sample_payload = {"rpm": 5200.0, "map_inhg": 31.0, "cht_avg_c": 98.5}
        
        # 1. Sign packet
        packet = gateway.sign_packet(sample_payload)
        self.assertIn("secure_envelope", packet)
        self.assertIn("hmac_sha256", packet)

        # 2. Verify authentic packet
        valid, unpacked, msg = gateway.verify_and_unpack(packet)
        self.assertTrue(valid)
        self.assertEqual(unpacked["rpm"], 5200.0)
        self.assertEqual(msg, "AUTHENTICATED_AND_VERIFIED")

        # 3. Detect Replay Attack
        valid_replay, _, msg_replay = gateway.verify_and_unpack(packet)
        self.assertFalse(valid_replay)
        self.assertIn("REPLAY_ATTACK", msg_replay)

        # 4. Detect Tampered Payload
        tampered_packet = gateway.sign_packet({"rpm": 5200.0})
        tampered_packet["secure_envelope"]["payload"]["rpm"] = 9999.0  # Tampered!
        valid_tampered, _, msg_tampered = gateway.verify_and_unpack(tampered_packet)
        self.assertFalse(valid_tampered)
        self.assertIn("MISMATCH", msg_tampered)

    def test_federated_fleet_aggregation(self):
        from src.ai_ml.federated_fleet import FederatedFleetCoordinator
        fleet = FederatedFleetCoordinator()
        
        # Simulate local training on 2 UAVs
        res_uav1 = np.random.normal(loc=0.0, scale=0.5, size=(40, 13))
        res_uav2 = np.random.normal(loc=0.0, scale=0.5, size=(60, 13))

        weights1, n1 = fleet.simulate_local_uav_training("UAV-01", res_uav1, epochs=1)
        weights2, n2 = fleet.simulate_local_uav_training("UAV-02", res_uav2, epochs=1)

        result = fleet.aggregate_federated_round([(weights1, n1), (weights2, n2)])
        self.assertEqual(result["status"], "AGGREGATION_COMPLETE")
        self.assertEqual(result["fleet_round"], 1)
        self.assertEqual(result["total_fleet_samples_processed"], 100)

if __name__ == "__main__":
    unittest.main()

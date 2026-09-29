"""
Flight Dataset Generator for DRDO Engine Simulation Demonstrator
Generates pre-recorded flight data sessions for all 5 MALE UAV mission profiles
for immediate out-of-the-box replay and post-flight health debriefing.
"""

import os
import json
import time
from pathlib import Path
from ..core.digital_twin_engine import DigitalTwinSynchronizer
from ..health.health_monitor import HealthMonitoringSystem
from ..diagnostics.fault_detector import IntelligentFaultDetector
from ..diagnostics.xai_explainer import XAIExplainer
from ..ai_ml.hybrid_detector import HybridPhysicsAIDetector
from ..ai_ml.rul_estimator import RULPrognosticsEstimator
from ..ai_ml.advisory_engine import MaintenanceAdvisoryEngine
from .fault_injector import FaultInjector
from .mission_simulator import MissionSimulator
from ..core.state import EngineFullStatePacket

def generate_datasets():
    recordings_dir = Path(__file__).parent.parent.parent / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    profiles = [
        "high_altitude_recon",
        "hot_desert_patrol",
        "rapid_throttle_tactical",
        "maritime_endurance",
        "fault_evaluation_flight"
    ]

    print("[*] Generating reference flight datasets for DRDO demonstrator...")

    for profile_key in profiles:
        injector = FaultInjector()
        sim = MissionSimulator(fault_injector=injector)
        sim.set_profile(profile_key)
        
        twin = DigitalTwinSynchronizer()
        health_mon = HealthMonitoringSystem()
        detector = IntelligentFaultDetector()
        xai = XAIExplainer()
        ai = HybridPhysicsAIDetector()
        rul = RULPrognosticsEstimator()
        advisory = MaintenanceAdvisoryEngine()

        output_file = recordings_dir / f"{profile_key}_sample.jsonl"
        print(f"    Simulating '{profile_key}' -> {output_file.name}...")

        with open(output_file, "w") as f:
            # Simulate 120 ticks at dt = 0.5s = 60 flight seconds
            for _ in range(120):
                telem = sim.step(dt_s=0.5)
                twin_est = twin.synchronize(telem)
                health_est = health_mon.evaluate(telem, twin_est)
                alerts = detector.detect(telem, twin_est)
                ai_res = ai.analyze(telem, twin_est)
                xai_res = xai.explain(telem, twin_est)
                prog = rul.update(health_est, dt_flight_hours=(0.5 / 3600.0))
                adv = advisory.generate_advisories(alerts, health_est, prog)

                packet = EngineFullStatePacket(
                    telemetry=telem,
                    digital_twin=twin_est,
                    health=health_est,
                    active_alerts=alerts,
                    prognosis=prog,
                    xai_attributions=xai_res.get("feature_attributions_pct", {})
                )
                f.write(json.dumps(packet.model_dump()) + "\n")

    print("[+] All 5 reference mission flight datasets generated successfully in recordings/")

if __name__ == "__main__":
    generate_datasets()

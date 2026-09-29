"""
DRDO Digital Twin Command-Line Analysis & Batch Processing Tool
Ingests flight data files (JSONL / CSV), executes digital twin state estimation,
evaluates all 8 fault modes, and outputs comprehensive DRDO mission debrief reports.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from .core.digital_twin_engine import DigitalTwinSynchronizer
from .health.health_monitor import HealthMonitoringSystem
from .diagnostics.fault_detector import IntelligentFaultDetector
from .diagnostics.xai_explainer import XAIExplainer
from .ai_ml.hybrid_detector import HybridPhysicsAIDetector
from .ai_ml.rul_estimator import RULPrognosticsEstimator
from .ai_ml.advisory_engine import MaintenanceAdvisoryEngine
from .reports.report_generator import DRDOMissionReportGenerator
from .core.state import TelemetryFrame, EngineFullStatePacket

def run_batch_analysis(input_filepath: str, output_dir: str = None) -> Dict[str, Any]:
    in_path = Path(input_filepath)
    if not in_path.exists():
        print(f"[!] Error: Input file '{input_filepath}' not found.")
        sys.exit(1)

    print("=" * 78)
    print("   DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)")
    print("   Aero Piston Engine Digital Twin - Batch Flight Analysis Tool")
    print(f"   Processing Sortie: {in_path.name}")
    print("=" * 78)

    twin = DigitalTwinSynchronizer()
    health_mon = HealthMonitoringSystem()
    detector = IntelligentFaultDetector()
    xai = XAIExplainer()
    ai = HybridPhysicsAIDetector()
    rul = RULPrognosticsEstimator()
    advisory = MaintenanceAdvisoryEngine()
    reporter = DRDOMissionReportGenerator(output_dir=output_dir)

    history: List[Dict[str, Any]] = []
    line_count = 0

    with open(in_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            data = json.loads(line)
            
            # Extract telemetry
            if "telemetry" in data:
                telem = TelemetryFrame(**data["telemetry"])
            else:
                telem = TelemetryFrame(**data)

            twin_est = twin.synchronize(telem)
            health_est = health_mon.evaluate(telem, twin_est)
            alerts = detector.detect(telem, twin_est)
            ai_res = ai.analyze(telem, twin_est)
            xai_res = xai.explain(telem, twin_est)
            prog = rul.update(health_est)
            adv = advisory.generate_advisories(alerts, health_est, prog)

            packet = EngineFullStatePacket(
                telemetry=telem,
                digital_twin=twin_est,
                health=health_est,
                active_alerts=alerts,
                prognosis=prog,
                xai_attributions=xai_res.get("feature_attributions_pct", {})
            )
            history.append(packet.model_dump())

    print(f"[*] Ingested {line_count} telemetry frames.")
    print("[*] Generating comprehensive mission health debrief...")
    report = reporter.generate_report(history, mission_name=in_path.stem.upper())

    print("\n" + "=" * 78)
    print("   MISSION HEALTH & RELIABILITY DEBRIEF SUMMARY")
    print("=" * 78)
    print(f"   Report ID:                  {report['report_id']}")
    print(f"   Sortie Duration:            {report['mission_duration_minutes']} minutes")
    print(f"   Final Engine Health:        {report['final_engine_health_index']}%")
    print(f"   Mission Reliability:        {report['mission_reliability_score']}%")
    print(f"   Remaining Useful Life (P50):{report['prognostics']['remaining_useful_life_p50_hours']} hours")
    print(f"   90% Confidence Interval:    {report['prognostics']['rul_confidence_interval_90pct']} hours")
    print(f"   Limiting Component:         {report['prognostics']['limiting_component']}")
    print(f"   Detected In-Flight Faults:  {report['detected_fault_count']}")
    
    if report["detected_faults"]:
        print("\n   Detected Incidents:")
        for idx, f in enumerate(report["detected_faults"], 1):
            print(f"     {idx}. [{f['severity']}] {f['fault_code']}: {f['title']}")
            print(f"        Action: {f['recommended_action']}")

    print("\n   Report Exports:")
    print(f"     HTML: {report['html_filepath']}")
    print(f"     JSON: {report['json_filepath']}")
    print("=" * 78)

    return report

def main():
    parser = argparse.ArgumentParser(description="DRDO MALE UAV Engine Digital Twin Offline Batch Ingestion Tool")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to telemetry dataset (JSONL / CSV)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory for reports")
    args = parser.parse_args()

    run_batch_analysis(args.input, args.output)

if __name__ == "__main__":
    main()

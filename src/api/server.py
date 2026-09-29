"""
FastAPI Server and Real-Time WebSocket Telemetry Engine
Provides real-time 10 Hz streaming of digital twin states, physics residuals,
XAI diagnostics, RUL prognostics, and REST endpoints for mission control.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse

from ..core.state import (
    TelemetryFrame,
    DigitalTwinEstimate,
    CompositeEngineHealth,
    RULPrognosis,
    EngineFullStatePacket
)
from ..core.digital_twin_engine import DigitalTwinSynchronizer
from ..core.bus import TelemetryBus, AeroCANProtocol
from ..health.health_monitor import HealthMonitoringSystem
from ..diagnostics.fault_detector import IntelligentFaultDetector
from ..diagnostics.xai_explainer import XAIExplainer
from ..ai_ml.hybrid_detector import HybridPhysicsAIDetector
from ..ai_ml.rul_estimator import RULPrognosticsEstimator
from ..ai_ml.advisory_engine import MaintenanceAdvisoryEngine
from ..simulation.fault_injector import FaultInjector
from ..simulation.mission_simulator import MissionSimulator
from ..simulation.mission_recorder import MissionRecorderReplay
from ..reports.report_generator import DRDOMissionReportGenerator
from ..physics.engine_maps import EnginePerformanceMaps
from ..core.sensor_fusion import EngineSensorFusionEKF
from ..health.mission_reliability_enhancer import MissionReliabilityEnhancer
from ..health.sensor_redundancy import AnalyticalSensorRedundancy

app = FastAPI(
    title="DRDO MALE UAV Aero Piston Engine Digital Twin System",
    description="Indigenous Digital Twin Core Framework for Real-Time Engine Health Monitoring and Predictive Maintenance (SIH26054)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Framework State Instances
fault_injector = FaultInjector()
simulator = MissionSimulator(fault_injector=fault_injector)
twin_engine = DigitalTwinSynchronizer()
health_monitor = HealthMonitoringSystem()
fault_detector = IntelligentFaultDetector()
xai_explainer = XAIExplainer()
hybrid_ai = HybridPhysicsAIDetector()
rul_estimator = RULPrognosticsEstimator()
advisory_engine = MaintenanceAdvisoryEngine()
recorder_replay = MissionRecorderReplay()
report_generator = DRDOMissionReportGenerator()
engine_maps = EnginePerformanceMaps()
sensor_fusion_ekf = EngineSensorFusionEKF()
reliability_enhancer = MissionReliabilityEnhancer()
sensor_redundancy_engine = AnalyticalSensorRedundancy()

# Telemetry session history buffer for report generation
telemetry_history: List[Dict[str, Any]] = []
history_max_len = 3600  # Up to 1 hour at 1 Hz downsampled or 6 mins at 10 Hz

# Active WebSocket connections
connected_websockets: List[WebSocket] = []

# Background simulation task handle
sim_task: Optional[asyncio.Task] = None
is_paused = False


async def telemetry_broadcast_loop():
    """Continuous 10 Hz simulation, digital twin sync, diagnostics, and streaming."""
    global is_paused, telemetry_history
    dt_s = 0.1
    tick_counter = 0

    while True:
        try:
            if not is_paused:
                # 1. Step simulation (or step replay if in replay mode)
                if recorder_replay.is_replaying:
                    frame_data = recorder_replay.step_replay()
                    if frame_data is None:
                        # Replay ended
                        recorder_replay.is_replaying = False
                        telem = simulator.step(dt_s=dt_s)
                    else:
                        telem = TelemetryFrame(**frame_data["telemetry"])
                else:
                    telem = simulator.step(dt_s=dt_s)

                # 2. Digital Twin Synchronization & Residual Computation
                twin_estimate = twin_engine.synchronize(telem)

                # 3. Health Monitoring & Subsystem Indices
                health = health_monitor.evaluate(telem, twin_estimate)

                # 4. Fault Detection & Predictive Diagnostics (8 fault modes)
                active_alerts = fault_detector.detect(telem, twin_estimate)

                # 5. Hybrid Physics-AI Anomaly Detection
                ai_anomaly = hybrid_ai.analyze(telem, twin_estimate)

                # 6. Explainable AI Feature Attribution
                xai_res = xai_explainer.explain(telem, twin_estimate)

                # 7. RUL Prognostics (Wiener Process + Bayesian Degradation)
                prognosis = rul_estimator.update(health, dt_flight_hours=(dt_s / 3600.0))

                # 8. Prescriptive Maintenance Advisories
                advisories = advisory_engine.generate_advisories(active_alerts, health, prognosis)

                # 9. Extended Kalman Filter (EKF) Sensor Fusion
                sensor_fusion_ekf.predict(telem.rpm, telem.throttle_pct, dt_s)
                ekf_state = sensor_fusion_ekf.update(telem)

                # 10. Mission Reliability Enhancement & Safe RTB Envelope
                reliability_envelope = reliability_enhancer.evaluate(telem, health)

                # 11. Thermodynamic Indicator P-V Cycle & Performance Maps
                pv_cycle = twin_engine.thermo_model.calculate_indicator_pv_cycle(
                    map_pa=telem.map_pa,
                    rpm=telem.rpm,
                    combustion_efficiency=0.98 if not active_alerts else 0.82
                )
                comp_point = engine_maps.evaluate_compressor_operating_point(
                    corrected_flow_kg_s=(telem.fuel_flow_lph * 14.2 * 0.72) / 3600.0,
                    pressure_ratio=telem.map_pa / max(10000.0, telem.ambient_pressure_pa)
                )

                # 12. Analytical Sensor Redundancy & Virtual Sensor Re-synthesis
                redundancy_status = sensor_redundancy_engine.evaluate_and_synthesize(telem, twin_estimate)
                fadec_state = {
                    "lane_a_status": "ACTIVE_MASTER",
                    "lane_b_status": "HOT_STANDBY",
                    "ccdl_bus": "STANAG_4586_SYNC",
                    "watchdog_ms": 1.2,
                    "sensor_voting_logic": "TRIPLE_MODULAR_REDUNDANCY",
                    "virtual_substitutions": redundancy_status.get("active_channels", []),
                    "operational_mode": "FAIL_OPERATIONAL_VIRTUAL_SYNTHESIS" if redundancy_status["virtual_substitutions_active"] else "DUAL_CHANNEL_NOMINAL"
                }

                # Assemble Full State Packet
                full_packet = EngineFullStatePacket(
                    telemetry=telem,
                    digital_twin=twin_estimate,
                    health=health,
                    active_alerts=active_alerts,
                    prognosis=prognosis,
                    xai_attributions=xai_res.get("feature_attributions_pct", {}),
                    sensor_fusion=ekf_state,
                    reliability_envelope=reliability_envelope,
                    sensor_redundancy=redundancy_status,
                    fadec_state=fadec_state
                )

                # Record frame if active
                if recorder_replay.active_session_file:
                    recorder_replay.record_frame(full_packet)

                # Append to downsampled history buffer for report generation (every 5 ticks = 2 Hz)
                tick_counter += 1
                if tick_counter % 5 == 0:
                    telemetry_history.append(full_packet.model_dump())
                    if len(telemetry_history) > history_max_len:
                        telemetry_history.pop(0)

                # Prepare payload for GCS WebSocket clients
                payload = {
                    "type": "TELEMETRY_UPDATE",
                    "telemetry": full_packet.telemetry.model_dump(),
                    "digital_twin": full_packet.digital_twin.model_dump(),
                    "health": full_packet.health.model_dump(),
                    "active_alerts": [a.model_dump() for a in full_packet.active_alerts],
                    "prognosis": full_packet.prognosis.model_dump(),
                    "ai_anomaly": ai_anomaly,
                    "xai": xai_res,
                    "advisories": advisories,
                    "sensor_fusion": ekf_state,
                    "reliability_envelope": reliability_envelope,
                    "pv_cycle": pv_cycle,
                    "compressor_point": comp_point,
                    "sensor_redundancy": redundancy_status,
                    "fadec_state": fadec_state,
                    "current_profile": simulator.current_profile_key,
                    "active_faults": fault_injector.get_state(),
                    "replay_status": recorder_replay.get_replay_progress()
                }

                # Broadcast to connected GCS dashboards
                disconnected = []
                for ws in connected_websockets:
                    try:
                        await ws.send_text(json.dumps(payload))
                    except Exception:
                        disconnected.append(ws)
                for ws in disconnected:
                    if ws in connected_websockets:
                        connected_websockets.remove(ws)

            await asyncio.sleep(dt_s)
        except asyncio.CancelledError:
            break
        except Exception as e:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    global sim_task
    # Start auto-recording default baseline
    recorder_replay.start_recording("auto_baseline_mission")
    sim_task = asyncio.create_task(telemetry_broadcast_loop())


@app.on_event("shutdown")
async def shutdown_event():
    global sim_task
    if sim_task:
        sim_task.cancel()


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            # Listen for client command messages
            data = await websocket.receive_text()
            cmd = json.loads(data)
            action = cmd.get("action")
            
            if action == "SET_PROFILE":
                simulator.set_profile(cmd.get("profile_key", "high_altitude_recon"))
            elif action == "INJECT_FAULT":
                fault_injector.inject_fault(
                    fault_type=cmd["fault_type"],
                    severity=cmd.get("severity", 0.6),
                    target_cylinder=cmd.get("target_cylinder", 2)
                )
            elif action == "CLEAR_FAULT":
                fault_injector.clear_fault(cmd.get("fault_type"))
            elif action == "CLEAR_ALL_FAULTS":
                fault_injector.clear_all()
            elif action == "THROTTLE_OVERRIDE":
                simulator.current_throttle_pct = max(0.0, min(100.0, float(cmd.get("throttle_pct", 70.0))))
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


# REST Endpoints
@app.get("/api/status")
async def get_status():
    return {
        "status": "ONLINE",
        "system": "DRDO MALE UAV Digital Twin Core",
        "current_profile": simulator.current_profile_key,
        "is_replaying": recorder_replay.is_replaying,
        "connected_clients": len(connected_websockets),
        "history_frames": len(telemetry_history)
    }


@app.get("/api/engine/maps")
async def get_engine_maps():
    """Returns calibrated 2D BSFC island mesh grid and aerodynamic compressor boundaries."""
    return {
        "bsfc_contour": engine_maps.get_bsfc_contour_data(),
        "compressor_limits": {
            "max_flow_kg_s": 0.16,
            "surge_line_equation": "PR = 1.0 + 12.8 * Flow_kg_s",
            "rated_boost_bar": 1.35
        }
    }


@app.get("/api/mission/reliability")
async def get_mission_reliability():
    """Returns current mission reliability assessment, derating envelope, and base reachability."""
    last_frame = simulator.step(dt_s=0.01)
    twin_est = twin_engine.synchronize(last_frame)
    health = health_monitor.evaluate(last_frame, twin_est)
    return reliability_enhancer.evaluate(last_frame, health)


@app.get("/api/mission/profiles")
async def list_profiles():
    return simulator.get_available_profiles()


@app.post("/api/mission/select")
async def select_profile(payload: Dict[str, Any]):
    key = payload.get("profile_key")
    if not key or key not in simulator.profiles:
        raise HTTPException(status_code=400, detail="Invalid profile key")
    simulator.set_profile(key)
    return {"status": "SUCCESS", "active_profile": key}


@app.post("/api/fault/inject")
async def inject_fault_endpoint(payload: Dict[str, Any]):
    fault_type = payload.get("fault_type")
    severity = payload.get("severity", 0.6)
    target_cylinder = payload.get("target_cylinder", 2)
    fault_injector.inject_fault(fault_type, severity, target_cylinder)
    return {"status": "FAULT_INJECTED", "active_faults": fault_injector.get_state()}


@app.post("/api/fault/clear")
async def clear_fault_endpoint(payload: Dict[str, Any]):
    fault_type = payload.get("fault_type")
    if fault_type == "ALL":
        fault_injector.clear_all()
    else:
        fault_injector.clear_fault(fault_type)
    return {"status": "FAULT_CLEARED", "active_faults": fault_injector.get_state()}


@app.post("/api/report/generate")
async def generate_report_endpoint():
    global telemetry_history
    if not telemetry_history:
        raise HTTPException(status_code=400, detail="Insufficient telemetry history to compile report")
    report = report_generator.generate_report(telemetry_history, mission_name=simulator.current_profile_key.upper())
    return report


@app.get("/api/recordings")
async def list_recordings():
    return recorder_replay.list_recorded_missions()


@app.post("/api/replay/load")
async def load_replay(payload: Dict[str, Any]):
    filename = payload.get("filename")
    success = recorder_replay.load_for_replay(filename)
    if not success:
        raise HTTPException(status_code=404, detail="Recording file not found")
    return {"status": "REPLAY_LOADED", "filename": filename, "total_frames": len(recorder_replay.replay_data)}


@app.post("/api/replay/scrub")
async def scrub_replay(payload: Dict[str, Any]):
    frame_idx = payload.get("frame_index", 0)
    frame = recorder_replay.scrub_to(frame_idx)
    return {"status": "SCRUBBED", "cursor": recorder_replay.replay_cursor, "frame": frame}


@app.post("/api/replay/exit")
async def exit_replay():
    recorder_replay.is_replaying = False
    return {"status": "EXITED_REPLAY"}


@app.get("/api/ml/metrics")
async def get_ml_metrics():
    """Returns verified offline ML training evaluation metrics and architecture dossier."""
    import os
    metrics_path = Path(__file__).parent.parent.parent / "reports" / "model_evaluation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "NOT_CALIBRATED",
        "message": "Run python src/ai_ml/train_models.py to generate calibration dossier"
    }


@app.get("/api/sensor_redundancy/status")
async def get_sensor_redundancy_status():
    """Returns real-time FADEC analytical sensor redundancy and virtual synthesis status."""
    return sensor_redundancy_engine.active_substitutions


@app.get("/api/fdr/export_csv")
async def export_fdr_csv():
    """Exports flight data recorder (FDR) telemetry history as a downloadable CSV."""
    global telemetry_history
    import io
    import csv
    from fastapi.responses import Response

    if not telemetry_history:
        # Generate dummy 10-line header if empty
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["timestamp_s", "mission_phase", "altitude_ft", "airspeed_kts", "rpm", "map_inhg", "cht_avg_c", "egt_avg_c", "oil_p_bar", "ehi", "rul_hours"])
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=DRDO_FDR_BLACKBOX.csv"})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp_s", "mission_phase", "altitude_ft", "airspeed_kts", "ambient_temp_c",
        "rpm", "throttle_pct", "map_inhg", "cht_1_c", "cht_2_c", "cht_3_c", "cht_4_c", "cht_avg_c",
        "egt_1_c", "egt_2_c", "egt_3_c", "egt_4_c", "egt_avg_c", "fuel_flow_lph", "fuel_pressure_bar",
        "oil_pressure_bar", "oil_temp_c", "vibration_rms_g", "crest_factor", "overall_health_index", "rul_hours_median"
    ])

    for frame in telemetry_history:
        t = frame.get("telemetry", {})
        h = frame.get("health", {})
        p = frame.get("prognosis", {})
        writer.writerow([
            t.get("timestamp_s", 0), t.get("mission_phase", ""), t.get("altitude_ft", 0), t.get("airspeed_kts", 0), t.get("ambient_temp_c", 0),
            t.get("rpm", 0), t.get("throttle_pct", 0), t.get("map_inhg", 0),
            t.get("cht_1_c", 0), t.get("cht_2_c", 0), t.get("cht_3_c", 0), t.get("cht_4_c", 0), t.get("cht_avg_c", 0),
            t.get("egt_1_c", 0), t.get("egt_2_c", 0), t.get("egt_3_c", 0), t.get("egt_4_c", 0), t.get("egt_avg_c", 0),
            t.get("fuel_flow_lph", 0), t.get("fuel_pressure_bar", 0),
            t.get("oil_pressure_bar", 0), t.get("oil_temp_c", 0),
            t.get("vibration_rms_g", 0), t.get("crest_factor", 0),
            h.get("overall_health_index", 1.0), p.get("rul_hours_median", 1800.0)
        ])

    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=DRDO_MALE_UAV_FDR_BLACKBOX.csv"})


# Mount Web GCS Dashboard Static Directory
web_dir = Path(__file__).parent.parent.parent / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/dashboard", StaticFiles(directory=str(web_dir), html=True), name="web")

# Mount Reports Directory for Direct PDF/HTML/JSON Viewing
reports_dir = Path(__file__).parent.parent.parent / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

@app.get("/")
async def root():
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "DRDO MALE UAV Digital Twin System Online. Dashboard at /dashboard"}

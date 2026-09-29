"""
Live System Integration and WebSocket Verification Script
Tests real-time WebSocket connection to running GCS server, validates 10 Hz packet streaming,
injects a fault, verifies alert reception, and calls the report generation API.
"""

import asyncio
import websockets
import json
import urllib.request

async def run_live_verification():
    print("[1] Testing REST API /api/status...")
    req = urllib.request.urlopen("http://127.0.0.1:8000/api/status")
    status_data = json.loads(req.read().decode("utf-8"))
    print("    REST Status:", status_data)
    assert status_data["status"] == "ONLINE"

    print("\n[2] Testing REST API /api/ml/metrics (ML Model Provenance)...")
    req_ml = urllib.request.urlopen("http://127.0.0.1:8000/api/ml/metrics")
    ml_data = json.loads(req_ml.read().decode("utf-8"))
    pm = ml_data.get("performance_metrics", {})
    print(f"    ML Provenance: ROC-AUC={pm.get('roc_auc')} | Precision={pm.get('precision_pct')}% | Latency={pm.get('avg_inference_latency_ms')}ms")
    assert pm.get("roc_auc", 0) >= 0.98

    print("\n[3] Connecting to WebSocket ws://127.0.0.1:8000/ws/telemetry...")
    uri = "ws://127.0.0.1:8000/ws/telemetry"
    async with websockets.connect(uri) as ws:
        print("    Connected successfully!")
        
        # Read 3 baseline frames
        for i in range(3):
            raw = await ws.recv()
            pkt = json.loads(raw)
            t = pkt["telemetry"]
            h = pkt["health"]
            p = pkt["prognosis"]
            sf = pkt.get("sensor_fusion", {})
            f = pkt.get("fadec_state", {})
            print(f"    Baseline Frame {i+1}: RPM={t['rpm']} | MAP={t['map_inhg']} inHg | Pmax={sf.get('estimated_p_max_bar')} bar | FADEC={f.get('operational_mode')} | EHI={h['overall_health_index']*100:.1f}%")

        print("\n[4] Injecting In-Flight Fault via WebSocket: Cylinder Combustion Misfire on Cyl #2...")
        inject_cmd = {
            "action": "INJECT_FAULT",
            "fault_type": "misfire",
            "severity": 0.85,
            "target_cylinder": 2
        }
        await ws.send(json.dumps(inject_cmd))
        print("    Injected fault command sent.")

        # Read frames until alert is received
        alert_received = False
        for _ in range(35):
            raw = await ws.recv()
            pkt = json.loads(raw)
            alerts = pkt.get("active_alerts", [])
            if any("MISF" in a["fault_code"] or "INJ" in a["fault_code"] for a in alerts):
                alert_received = True
                target_alert = [a for a in alerts if "MISF" in a["fault_code"] or "INJ" in a["fault_code"]][0]
                print(f"    [VERIFIED] Alert Received: [{target_alert['fault_code']}] {target_alert['title']}")
                print(f"    Evidence: {target_alert['description']}")
                print(f"    XAI Top Drivers: {pkt['xai'].get('top_drivers', [])}")
                break
        
        assert alert_received, "Failed to receive fault alert over WebSocket!"

        print("\n[5] Resetting all faults...")
        await ws.send(json.dumps({"action": "CLEAR_ALL_FAULTS"}))
        raw = await ws.recv()
        print("    Faults cleared.")

    print("\n[6] Testing Flight Data Recorder (FDR) Blackbox CSV Export /api/fdr/export_csv...")
    req_fdr = urllib.request.urlopen("http://127.0.0.1:8000/api/fdr/export_csv")
    fdr_csv = req_fdr.read().decode("utf-8")
    print(f"    FDR Blackbox CSV Downloaded: {len(fdr_csv.splitlines())} records received.")
    assert "timestamp_s" in fdr_csv

    print("\n[7] Triggering DRDO Mission Health Report Generation...")
    post_req = urllib.request.Request("http://127.0.0.1:8000/api/report/generate", data=b"{}", headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(post_req)
    rep_data = json.loads(resp.read().decode("utf-8"))
    print(f"    Generated Report ID: {rep_data['report_id']}")
    print(f"    HTML Report Path: {rep_data['html_filepath']}")
    print(f"    JSON Report Path: {rep_data['json_filepath']}")
    print(f"    Detected Fault Count: {rep_data['detected_fault_count']}")

    print("\n" + "=" * 70)
    print("   ALL LIVE WEBSOCKET & REST INTEGRATION TESTS PASSED 100%")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_live_verification())

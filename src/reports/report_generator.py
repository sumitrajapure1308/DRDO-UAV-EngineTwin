"""
Mission Health Report and Engineering Debrief Generator
Generates comprehensive DRDO-standard mission reports summarizing engine health metrics,
thermal profiles, lubrication status, anomaly history, RUL consumption, and maintenance work orders.
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Any
from ..core.state import EngineFullStatePacket

class DRDOMissionReportGenerator:
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "reports")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, history: List[Dict[str, Any]], mission_name: str = "MALE_UAV_ISR_SORTIE") -> Dict[str, Any]:
        """
        Analyze a mission packet sequence and produce a comprehensive technical health report.
        """
        if not history:
            return {"error": "No telemetry data provided"}

        start_time = history[0]["telemetry"]["timestamp_s"]
        end_time = history[-1]["telemetry"]["timestamp_s"]
        duration_s = max(1.0, end_time - start_time)

        # Statistics computation
        chts_1 = [h["telemetry"]["cht_1_c"] for h in history]
        chts_2 = [h["telemetry"]["cht_2_c"] for h in history]
        chts_3 = [h["telemetry"]["cht_3_c"] for h in history]
        chts_4 = [h["telemetry"]["cht_4_c"] for h in history]
        all_chts = chts_1 + chts_2 + chts_3 + chts_4

        egts_1 = [h["telemetry"]["egt_1_c"] for h in history]
        egts_2 = [h["telemetry"]["egt_2_c"] for h in history]
        egts_3 = [h["telemetry"]["egt_3_c"] for h in history]
        egts_4 = [h["telemetry"]["egt_4_c"] for h in history]
        all_egts = egts_1 + egts_2 + egts_3 + egts_4

        oil_ps = [h["telemetry"]["oil_pressure_bar"] for h in history]
        oil_ts = [h["telemetry"]["oil_temp_c"] for h in history]
        vibs = [h["telemetry"]["vibration_rms_g"] for h in history]
        ehi_vals = [h["health"]["overall_health_index"] for h in history]

        # Collect unique alerts triggered during mission
        unique_alerts: Dict[str, Dict[str, Any]] = {}
        for h in history:
            for alt in h.get("active_alerts", []):
                key = f"{alt['fault_code']}_{alt.get('affected_cylinder', 0)}"
                if key not in unique_alerts:
                    unique_alerts[key] = alt

        final_packet = history[-1]
        final_health = final_packet["health"]
        final_prognosis = final_packet["prognosis"]

        report_summary = {
            "report_id": f"DRDO-EHR-{int(time.time())}",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "organization": "DRDO Department of Defence R&D - Theme: Robotics & Drones",
            "problem_statement_id": "SIH26054",
            "mission_name": mission_name,
            "mission_duration_minutes": round(duration_s / 60.0, 1),
            "mission_phases_covered": list(set(h["telemetry"]["mission_phase"] for h in history)),
            
            # Health indices
            "initial_engine_health_index": round(ehi_vals[0] * 100.0, 1),
            "minimum_engine_health_index": round(min(ehi_vals) * 100.0, 1),
            "final_engine_health_index": round(ehi_vals[-1] * 100.0, 1),
            "health_degradation_pct": round((ehi_vals[0] - ehi_vals[-1]) * 100.0, 2),
            "mission_reliability_score": final_health["mission_reliability_pct"],
            
            # Subsystem scores at mission end
            "final_subsystem_health": final_health["subsystem_scores"],
            
            # Thermal Extremes
            "max_cht_c": round(max(all_chts), 1),
            "max_egt_c": round(max(all_egts), 1),
            "max_cht_spread_c": round(max(h["telemetry"]["cht_spread_c"] for h in history), 1),
            "max_egt_spread_c": round(max(h["telemetry"]["egt_spread_c"] for h in history), 1),
            
            # Lubrication Extremes
            "min_oil_pressure_bar": round(min(oil_ps), 2),
            "max_oil_temp_c": round(max(oil_ts), 1),
            
            # Vibration
            "max_vibration_rms_g": round(max(vibs), 2),
            
            # Fault log
            "detected_fault_count": len(unique_alerts),
            "detected_faults": list(unique_alerts.values()),
            
            # Prognostics & Maintenance Advisories
            "prognostics": {
                "accumulated_flight_hours": final_prognosis["current_operating_hours"],
                "remaining_useful_life_p50_hours": final_prognosis["rul_hours_median"],
                "rul_confidence_interval_90pct": [final_prognosis["rul_hours_p10"], final_prognosis["rul_hours_p90"]],
                "limiting_component": final_prognosis["primary_limiting_subsystem"],
                "degradation_trend": final_prognosis["degradation_trend"]
            }
        }

        # Save JSON report
        json_file = self.output_dir / f"health_report_{int(time.time())}.json"
        with open(json_file, "w") as f:
            json.dump(report_summary, f, indent=2)

        # Generate HTML report
        html_file = self.output_dir / f"health_report_{int(time.time())}.html"
        html_content = self._render_html_report(report_summary)
        with open(html_file, "w") as f:
            f.write(html_content)

        report_summary["json_filepath"] = str(json_file)
        report_summary["html_filepath"] = str(html_file)
        return report_summary

    def _render_html_report(self, r: Dict[str, Any]) -> str:
        fault_rows = ""
        for f in r["detected_faults"]:
            sev_class = "danger" if f["severity"] == "CRITICAL" else "warning"
            fault_rows += f"""
            <tr>
                <td><code>{f['fault_code']}</code></td>
                <td><span class="badge {sev_class}">{f['severity']}</span></td>
                <td><strong>{f['title']}</strong></td>
                <td>{f['description']}</td>
                <td><em>{f['recommended_action']}</em></td>
            </tr>
            """
        if not fault_rows:
            fault_rows = "<tr><td colspan='5' class='text-center'>No abnormal faults detected during this sortie.</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRDO Aero Engine Mission Health Report - {r['report_id']}</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0c1017; color: #c9d1d9; margin: 0; padding: 25px; line-height: 1.5; }}
    .report-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 25px; max-width: 1000px; margin: 0 auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
    .header {{ border-bottom: 2px solid #238636; padding-bottom: 15px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }}
    h1 {{ color: #58a6ff; margin: 0; font-size: 24px; }}
    .meta-tag {{ color: #8b949e; font-size: 13px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
    .kpi-box {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 15px; text-align: center; }}
    .kpi-val {{ font-size: 24px; font-weight: bold; color: #3fb950; margin: 5px 0; }}
    .kpi-val.warning {{ color: #d29922; }}
    .kpi-val.danger {{ color: #f85149; }}
    .kpi-lbl {{ font-size: 11px; text-transform: uppercase; color: #8b949e; letter-spacing: 0.5px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
    th, td {{ border: 1px solid #30363d; padding: 10px; text-align: left; }}
    th {{ background: #21262d; color: #c9d1d9; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }}
    .badge.danger {{ background: #f8514933; color: #f85149; border: 1px solid #f85149; }}
    .badge.warning {{ background: #d2992233; color: #d29922; border: 1px solid #d29922; }}
    .section-title {{ color: #79c0ff; font-size: 16px; margin-top: 25px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }}
</style>
</head>
<body>
<div class="report-card">
    <div class="header">
        <div>
            <h1>DEFENCE RESEARCH &amp; DEVELOPMENT ORGANISATION</h1>
            <div class="meta-tag">MALE UAV Aero Piston Engine Digital Twin Health &amp; Reliability Debrief</div>
            <div class="meta-tag">Report ID: <strong>{r['report_id']}</strong> | Generated: {r['generated_at']}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 18px; font-weight: bold; color: #3fb950;">DRDO SIH26054</div>
            <div class="meta-tag">Mission: {r['mission_name']}</div>
            <div class="meta-tag">Duration: {r['mission_duration_minutes']} min</div>
        </div>
    </div>

    <div class="kpi-grid">
        <div class="kpi-box">
            <div class="kpi-lbl">Final Engine Health</div>
            <div class="kpi-val {'danger' if r['final_engine_health_index'] < 60 else 'warning' if r['final_engine_health_index'] < 85 else ''}">{r['final_engine_health_index']}%</div>
            <div class="meta-tag">Reliability: {r['mission_reliability_score']}%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-lbl">Remaining Useful Life (P50)</div>
            <div class="kpi-val">{r['prognostics']['remaining_useful_life_p50_hours']} hrs</div>
            <div class="meta-tag">TBO Confidence: 90%</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-lbl">Peak Cylinder Head Temp</div>
            <div class="kpi-val {'danger' if r['max_cht_c'] > 135 else ''}">{r['max_cht_c']} °C</div>
            <div class="meta-tag">Max Spread: {r['max_cht_spread_c']} °C</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-lbl">Active Faults Logged</div>
            <div class="kpi-val {'danger' if r['detected_fault_count'] > 0 else ''}">{r['detected_fault_count']}</div>
            <div class="meta-tag">Limiting: {r['prognostics']['limiting_component'].split('/')[0]}</div>
        </div>
    </div>

    <div class="section-title">ENGINE SUB-SYSTEM HEALTH EVALUATION</div>
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px;">
        <div class="kpi-box"><div class="kpi-lbl">Thermal CHT</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['thermal_cht']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Exhaust Gas Path</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['exhaust_gas_path']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Fuel Injection</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['fuel_injection']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Lubrication System</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['lubrication_system']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Turbocharger Boost</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['turbo_charge']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Cooling Loop</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['cooling_system']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Combustion Stability</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['combustion_stability']}%</div></div>
        <div class="kpi-box"><div class="kpi-lbl">Electrical Bus</div><div class="kpi-val" style="font-size:18px;">{r['final_subsystem_health']['electrical_bus']}%</div></div>
    </div>

    <div class="section-title">INTELLIGENT FAULT &amp; PREDICTIVE INCIDENT LOG</div>
    <table>
        <thead>
            <tr>
                <th>ATA Code</th>
                <th>Severity</th>
                <th>Diagnostic Title</th>
                <th>Physical Evidence &amp; Symptoms</th>
                <th>Prescribed Maintenance Procedure</th>
            </tr>
        </thead>
        <tbody>
            {fault_rows}
        </tbody>
    </table>

    <div class="section-title">PRESCRIPTIVE PROGNOSTIC MAINTENANCE DIRECTIVE</div>
    <div style="background: #0d1117; border-left: 4px solid #58a6ff; padding: 15px; margin-top: 10px; border-radius: 4px;">
        <p><strong>Primary Limiting Subsystem:</strong> {r['prognostics']['limiting_component']}</p>
        <p><strong>Degradation Trend:</strong> <code>{r['prognostics']['degradation_trend']}</code></p>
        <p><strong>Confidence Interval (90%):</strong> P10 = {r['prognostics']['rul_confidence_interval_90pct'][0]} hrs, P50 = {r['prognostics']['remaining_useful_life_p50_hours']} hrs, P90 = {r['prognostics']['rul_confidence_interval_90pct'][1]} hrs.</p>
        <p><strong>Ground Crew Action:</strong> Review prescribed maintenance procedures in the incident log above prior to next scheduled sortie.</p>
    </div>
</div>
</body>
</html>"""

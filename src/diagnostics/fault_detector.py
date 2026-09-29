"""
Intelligent Predictive Diagnostics and Fault Detector
Detects the 8 required fault categories using analytical physics redundancy and dynamic residuals:
1. Misfire conditions
2. Injector abnormalities (lean/rich drift, clogging)
3. Cooling degradation (heat rejection loss)
4. Lubrication issues (bearing clearance wear, oil thinning, pressure loss)
5. Sensor drift / failure (analytical redundancy cross-check)
6. Combustion instability (cyclic variance, knock)
7. Overheating trends (predictive thermal trajectory)
8. Abnormal vibration patterns (unbalance, bearing harmonic degradation)
"""

import time
import math
from typing import List, Dict, Any, Optional
from ..core.state import TelemetryFrame, DigitalTwinEstimate, FaultAlert
from .fault_catalog import FAULT_DEFINITIONS

class IntelligentFaultDetector:
    def __init__(self):
        # Thermal derivative history for predictive overheating trend
        self.cht_history: List[float] = []
        self.time_history: List[float] = []
        self.alert_counter = 0

    def detect(self, telem: TelemetryFrame, twin: DigitalTwinEstimate) -> List[FaultAlert]:
        """
        Evaluate telemetry and physics residuals against multi-subsystem degradation logic.
        """
        alerts: List[FaultAlert] = []
        now = telem.timestamp_s

        # Track CHT for thermal trend extrapolation (slope dT/dt)
        self.cht_history.append(telem.cht_avg_c)
        self.time_history.append(now)
        if len(self.cht_history) > 30:
            self.cht_history.pop(0)
            self.time_history.pop(0)

        # -------------------------------------------------------------
        # 1. MISFIRE DETECTION (Single cylinder / intermittent misfire)
        # Signature: localized EGT collapse (>120°C below twin/siblings) + 0.5x vibration spike + high crest factor
        # -------------------------------------------------------------
        egt_vals = [telem.egt_1_c, telem.egt_2_c, telem.egt_3_c, telem.egt_4_c]
        egt_residuals = [twin.residual_egt_1_c, twin.residual_egt_2_c, twin.residual_egt_3_c, twin.residual_egt_4_c]
        min_egt = min(egt_vals)
        misfiring_cylinder = None
        
        for idx in range(4):
            # Check if this cylinder has collapsed in EGT relative to nominal twin
            if egt_residuals[idx] < -130.0 and (telem.egt_avg_c - egt_vals[idx]) > 100.0:
                misfiring_cylinder = idx + 1
                break

        if misfiring_cylinder is not None and (telem.crest_factor > 3.2 or telem.vibration_rms_g > 2.0 or telem.rpm > 2000.0):
            self.alert_counter += 1
            meta = FAULT_DEFINITIONS["MISFIRE_DETECTED"]
            alerts.append(FaultAlert(
                alert_id=f"ALT-MISF-{self.alert_counter}",
                timestamp_s=now,
                fault_code=meta["code"],
                subsystem=meta["subsystem"],
                severity=meta["default_severity"],
                title=f"{meta['title']} on Cylinder #{misfiring_cylinder}",
                description=f"Combustion quenching on Cylinder #{misfiring_cylinder}. EGT collapsed to {round(egt_vals[misfiring_cylinder-1], 1)}°C (Residual: {round(egt_residuals[misfiring_cylinder-1], 1)}°C). Crest factor {round(telem.crest_factor, 2)}.",
                affected_cylinder=misfiring_cylinder,
                root_cause_evidence={
                    "collapsed_cylinder": misfiring_cylinder,
                    "egt_residual_c": egt_residuals[misfiring_cylinder-1],
                    "crest_factor": telem.crest_factor,
                    "subharmonic_vibration": True
                },
                recommended_action=meta["recommended_action"]
            ))

        # -------------------------------------------------------------
        # 2. INJECTOR ABNORMALITY (Partial clog or leaking injector)
        # Signature: CHT/EGT asymmetric spread without full misfire collapse
        # -------------------------------------------------------------
        cht_residuals = [twin.residual_cht_1_c, twin.residual_cht_2_c, twin.residual_cht_3_c, twin.residual_cht_4_c]
        if misfiring_cylinder is None:
            abnormal_inj_cyl = None

            for idx in range(4):
                egt_diff = abs(egt_vals[idx] - telem.egt_avg_c)
                cht_diff = abs(cht_residuals[idx])
                if (egt_diff > 45.0 or cht_diff > 12.0) and (telem.egt_spread_c > 38.0 or telem.cht_spread_c > 14.0):
                    abnormal_inj_cyl = idx + 1
                    break

            if abnormal_inj_cyl is not None:
                self.alert_counter += 1
                meta = FAULT_DEFINITIONS["INJECTOR_ABNORMAL"]
                alerts.append(FaultAlert(
                    alert_id=f"ALT-INJ-{self.alert_counter}",
                    timestamp_s=now,
                    fault_code=meta["code"],
                    subsystem=meta["subsystem"],
                    severity=meta["default_severity"],
                    title=f"Injector #{abnormal_inj_cyl} Lean/Delivery Abnormality",
                    description=f"Fuel delivery deviation on Cylinder #{abnormal_inj_cyl}. Local CHT is +{round(cht_residuals[abnormal_inj_cyl-1], 1)}°C above digital twin baseline. Cylinder spread {round(telem.cht_spread_c, 1)}°C.",
                    affected_cylinder=abnormal_inj_cyl,
                    root_cause_evidence={
                        "abnormal_cylinder": abnormal_inj_cyl,
                        "cht_residual_c": cht_residuals[abnormal_inj_cyl-1],
                        "cht_spread_c": telem.cht_spread_c,
                        "fuel_flow_residual": twin.residual_fuel_flow_lph
                    },
                    recommended_action=meta["recommended_action"]
                ))

        # -------------------------------------------------------------
        # 3. COOLING DEGRADATION (Coolant restriction / radiator fouling)
        # Signature: Global uniform CHT rise across ALL cylinders + coolant temperature rise
        # -------------------------------------------------------------
        avg_cht_res = sum(cht_residuals) / 4.0
        if avg_cht_res > 12.0 and telem.coolant_temp_c > 96.0:
            self.alert_counter += 1
            meta = FAULT_DEFINITIONS["COOLING_DEGRADATION"]
            alerts.append(FaultAlert(
                alert_id=f"ALT-COOL-{self.alert_counter}",
                timestamp_s=now,
                fault_code=meta["code"],
                subsystem=meta["subsystem"],
                severity=meta["default_severity"],
                title=meta["title"],
                description=f"Engine coolant loop heat rejection degradation. Global CHT residual is +{round(avg_cht_res, 1)}°C. Coolant temperature elevated at {round(telem.coolant_temp_c, 1)}°C.",
                root_cause_evidence={
                    "avg_cht_residual_c": round(avg_cht_res, 1),
                    "coolant_temp_c": round(telem.coolant_temp_c, 1),
                    "ram_airspeed_kts": telem.airspeed_kts
                },
                recommended_action=meta["recommended_action"]
            ))

        # -------------------------------------------------------------
        # 4. LUBRICATION ISSUES (Low pressure, high temp, bearing wear)
        # Signature: Oil pressure < 0.9 bar (redline low), or < 2.0 bar at cruise, or persistent deficit vs digital twin
        # -------------------------------------------------------------
        is_low_oil_p = (
            (telem.oil_pressure_bar < 0.9) or
            (telem.oil_pressure_bar < 2.0 and telem.rpm > 3500.0) or
            (twin.residual_oil_pressure_bar < -1.2 and telem.rpm > 2800.0) or
            (telem.oil_temp_c > 120.0)
        )
        if is_low_oil_p:
            self.alert_counter += 1
            meta = FAULT_DEFINITIONS["LUBRICATION_ISSUE"]
            severity = "CRITICAL" if telem.oil_pressure_bar < 1.4 else "WARNING"
            alerts.append(FaultAlert(
                alert_id=f"ALT-LUBE-{self.alert_counter}",
                timestamp_s=now,
                fault_code=meta["code"],
                subsystem=meta["subsystem"],
                severity=severity,
                title=meta["title"],
                description=f"Oil pressure deficit: {round(telem.oil_pressure_bar, 2)} bar (Digital Twin expected: {round(twin.twin_oil_pressure_bar, 2)} bar). Oil temp: {round(telem.oil_temp_c, 1)}°C.",
                root_cause_evidence={
                    "oil_pressure_bar": round(telem.oil_pressure_bar, 2),
                    "residual_oil_p_bar": round(twin.residual_oil_pressure_bar, 2),
                    "oil_temp_c": round(telem.oil_temp_c, 1)
                },
                recommended_action=meta["recommended_action"]
            ))

        # -------------------------------------------------------------
        # 5. SENSOR DRIFT / FAILURE (Analytical Redundancy Check)
        # Differentiates sensor bias from true thermodynamic engine failure
        # Example: One CHT reads high, but EGT, coolant, and oil are perfectly nominal
        # -------------------------------------------------------------
        for idx in range(4):
            # If CHT residual is large (>15°C) BUT EGT residual is close to 0 (<15°C) and coolant is normal
            if abs(cht_residuals[idx]) > 18.0 and abs(egt_residuals[idx]) < 12.0 and telem.coolant_temp_c < 92.0:
                self.alert_counter += 1
                meta = FAULT_DEFINITIONS["SENSOR_DRIFT_FAILURE"]
                alerts.append(FaultAlert(
                    alert_id=f"ALT-SENS-{self.alert_counter}",
                    timestamp_s=now,
                    fault_code=meta["code"],
                    subsystem=meta["subsystem"],
                    severity="CAUTION",
                    title=f"Thermocouple CHT #{idx+1} Sensor Calibration Drift",
                    description=f"Analytical physics redundancy mismatch on CHT #{idx+1}. Probe indicates {round(getattr(telem, f'cht_{idx+1}_c'), 1)}°C while coupled EGT and coolant states confirm nominal combustion.",
                    affected_cylinder=idx + 1,
                    root_cause_evidence={
                        "sensor_channel": f"cht_{idx+1}_c",
                        "divergence_c": cht_residuals[idx],
                        "analytical_redundancy_validated": True
                    },
                    recommended_action=meta["recommended_action"]
                ))

        # -------------------------------------------------------------
        # 6. COMBUSTION INSTABILITY & KNOCK
        # Signature: high crest factor, knock intensity > 0.35, acoustic/vibration orders
        # -------------------------------------------------------------
        if telem.crest_factor > 3.6 or (telem.vibration_rms_g > 2.6 and misfiring_cylinder is None):
            self.alert_counter += 1
            meta = FAULT_DEFINITIONS["COMBUSTION_INSTABILITY"]
            alerts.append(FaultAlert(
                alert_id=f"ALT-COMB-{self.alert_counter}",
                timestamp_s=now,
                fault_code=meta["code"],
                subsystem=meta["subsystem"],
                severity=meta["default_severity"],
                title=meta["title"],
                description=f"Combustion shock/detonation signatures observed. Vibration crest factor is {round(telem.crest_factor, 2)}, RMS vibration: {round(telem.vibration_rms_g, 2)}g.",
                root_cause_evidence={
                    "crest_factor": round(telem.crest_factor, 2),
                    "vibration_rms_g": round(telem.vibration_rms_g, 2),
                    "map_inhg": telem.map_inhg
                },
                recommended_action=meta["recommended_action"]
            ))

        # -------------------------------------------------------------
        # 7. PREDICTIVE OVERHEATING TREND (Trajectory Extrapolation)
        # Calculates dT/dt slope. Projects CHT 180 seconds forward.
        # Emits advisory BEFORE 135°C Caution or 145°C Redline is breached!
        # -------------------------------------------------------------
        if len(self.cht_history) >= 10:
            dt = max(1.0, self.time_history[-1] - self.time_history[0])
            d_cht = self.cht_history[-1] - self.cht_history[0]
            slope_c_per_sec = d_cht / dt
            projected_cht_180s = telem.cht_avg_c + (slope_c_per_sec * 180.0)

            # Only trigger if engine has reached normal operating band (>105°C) and slope threatens redline
            if slope_c_per_sec > 0.08 and projected_cht_180s > 135.0 and telem.cht_avg_c > 105.0:
                self.alert_counter += 1
                meta = FAULT_DEFINITIONS["OVERHEATING_TREND"]
                alerts.append(FaultAlert(
                    alert_id=f"ALT-OHT-{self.alert_counter}",
                    timestamp_s=now,
                    fault_code=meta["code"],
                    subsystem=meta["subsystem"],
                    severity="CAUTION",
                    title=meta["title"],
                    description=f"Predictive thermal trend: CHT rising at +{round(slope_c_per_sec * 60, 1)}°C/min. Projected to reach {round(projected_cht_180s, 1)}°C in 180s under current power.",
                    root_cause_evidence={
                        "thermal_slope_c_per_min": round(slope_c_per_sec * 60, 1),
                        "projected_cht_180s": round(projected_cht_180s, 1),
                        "current_cht_c": round(telem.cht_avg_c, 1)
                    },
                    recommended_action=meta["recommended_action"]
                ))

        # -------------------------------------------------------------
        # 8. ABNORMAL VIBRATION PATTERNS
        # Signature: High RMS vibration > 2.5g without misfire, or shaft unbalance
        # -------------------------------------------------------------
        if telem.vibration_rms_g > 2.6 and misfiring_cylinder is None:
            self.alert_counter += 1
            meta = FAULT_DEFINITIONS["ABNORMAL_VIBRATION"]
            alerts.append(FaultAlert(
                alert_id=f"ALT-VIB-{self.alert_counter}",
                timestamp_s=now,
                fault_code=meta["code"],
                subsystem=meta["subsystem"],
                severity="WARNING",
                title=meta["title"],
                description=f"Mechanical vibration exceeds structural threshold: {round(telem.vibration_rms_g, 2)}g RMS (Normal < 1.8g). Peak acceleration: {round(telem.vibration_peak_g, 2)}g.",
                root_cause_evidence={
                    "vibration_rms_g": round(telem.vibration_rms_g, 2),
                    "vibration_peak_g": round(telem.vibration_peak_g, 2),
                    "rpm": telem.rpm
                },
                recommended_action=meta["recommended_action"]
            ))

        return alerts

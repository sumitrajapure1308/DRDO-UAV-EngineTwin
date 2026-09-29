"""
DRDO MALE UAV Aero Piston Engine Digital Twin
Mission Reliability Enhancer & Tactical Flight Envelope Reconfiguration Engine

Implements autonomous operational decision logic to maximize UAV mission survival:
- Dynamic Throttle Derating to arrest progressive mechanical/thermal failure
- Safe Return-To-Base (RTB) Powered & Gliding Range Envelope (Nautical Miles)
- Tactical Airfield Diversion Assessment (Home Base vs Forward Operating Bases)
- Go / No-Go Decision Logic for Secondary ISR Mission Tasking
"""

import math
from typing import Dict, Any, List
from ..core.state import TelemetryFrame, CompositeEngineHealth, SubsystemHealth

class MissionReliabilityEnhancer:
    """
    Evaluates real-time engine degradation and computes dynamic flight envelope constraints
    to maximize MALE UAV asset preservation and mission completion probability.
    """

    def __init__(self):
        # Aerodynamic and operational parameters for typical MALE UAV (e.g. TAPAS-BH-201 class)
        self.lift_to_drag_ratio = 18.2          # Clean gliding ratio
        self.nominal_cruise_speed_kts = 110.0   # True airspeed
        self.fuel_capacity_kg = 85.0            # Total fuel tankage
        
        # Strategic recovery bases (distance in Nautical Miles from current combat patrol area)
        self.recovery_bases = [
            {"name": "Home Base (AFB Main)", "distance_nm": 95.0, "runway_m": 2800, "priority": 1},
            {"name": "Forward Operating Base (FOB-Alpha)", "distance_nm": 42.0, "runway_m": 1600, "priority": 2},
            {"name": "Emergency Tactical Strip (EHS-Bravo)", "distance_nm": 18.0, "runway_m": 900, "priority": 3}
        ]

    def evaluate(
        self,
        telem: TelemetryFrame,
        health: CompositeEngineHealth,
        fuel_remaining_liters: float = 48.0
    ) -> Dict[str, Any]:
        """
        Evaluate engine health deficit and determine in-flight derating limits and RTB envelope.
        """
        ehi = max(0.05, min(1.0, health.overall_health_index))
        sub = health.subsystem_scores

        fuel_kg = max(2.0, float(fuel_remaining_liters) * 0.72)
        ff_kg_hr = max(3.0, telem.fuel_flow_lph * 0.72)
        tas_kts = max(60.0, telem.airspeed_kts if telem.airspeed_kts > 30 else self.nominal_cruise_speed_kts)
        alt_ft = max(500.0, telem.altitude_ft)

        # 1. Compute Safe Throttle Derating Cap
        max_safe_throttle_pct = 100.0
        derate_reasons = []

        # Check Oil Pressure degradation
        if telem.oil_pressure_bar < 2.0:
            max_safe_throttle_pct = min(max_safe_throttle_pct, 45.0)
            derate_reasons.append("Low Oil Pressure: Cap power to 45% to prevent hydrodynamic bearing seizure")
        elif telem.oil_pressure_bar < 2.8:
            max_safe_throttle_pct = min(max_safe_throttle_pct, 65.0)
            derate_reasons.append("Oil Pressure Deficit: Restrict power to 65%")

        # Check Cylinder Head Overheating
        max_cht = max(telem.cht_1_c, telem.cht_2_c, telem.cht_3_c, telem.cht_4_c)
        if max_cht > 140.0:
            max_safe_throttle_pct = min(max_safe_throttle_pct, 55.0)
            derate_reasons.append(f"Critical CHT ({max_cht:.0f}°C): Derate to 55% for thermal stabilization")
        elif max_cht > 125.0:
            max_safe_throttle_pct = min(max_safe_throttle_pct, 75.0)
            derate_reasons.append(f"Elevated CHT ({max_cht:.0f}°C): Cap climb throttle to 75%")

        # Check Vibration unbalance
        if telem.vibration_rms_g > 3.0:
            max_safe_throttle_pct = min(max_safe_throttle_pct, 50.0)
            derate_reasons.append("Severe Vibration (>3.0g): Slew throttle away from harmonic resonance")

        # 2. Powered & Gliding Range Calculation
        # Hours of powered endurance remaining at current consumption
        powered_endurance_hrs = fuel_kg / ff_kg_hr
        powered_range_nm = powered_endurance_hrs * tas_kts

        # Dead-stick gliding range (Glide Ratio * Altitude)
        glide_range_nm = (alt_ft / 6076.12) * self.lift_to_drag_ratio

        total_reachable_range_nm = powered_range_nm + glide_range_nm

        # 3. Tactical Base Reachability Matrix
        base_status = []
        best_divert_base = None

        for b in self.recovery_bases:
            dist = b["distance_nm"]
            margin_nm = total_reachable_range_nm - dist
            reachable = margin_nm > 0.0
            fuel_at_arrival_kg = max(0.0, fuel_kg - (dist / tas_kts) * ff_kg_hr)

            status_entry = {
                "base_name": b["name"],
                "distance_nm": dist,
                "reachable": reachable,
                "safety_margin_nm": round(margin_nm, 1),
                "est_fuel_reserve_at_touchdown_kg": round(fuel_at_arrival_kg, 1)
            }
            base_status.append(status_entry)

            if reachable and best_divert_base is None and ehi < 0.60:
                best_divert_base = b["name"]

        # 4. Mission Operational Directive
        if ehi >= 0.88 and max_safe_throttle_pct >= 95.0:
            mission_directive = "GO_FULL_MISSION"
            operational_advisory = "Engine propulsion fully nominal. Maintain scheduled mission trajectory and sensor payload tasking."
        elif ehi >= 0.70:
            mission_directive = "GO_WITH_ENVELOPE_RESTRICTION"
            operational_advisory = f"Subsystem stress detected. Cap throttle to {max_safe_throttle_pct:.0f}%. Avoid maximum rate climbs."
        elif ehi >= 0.45:
            mission_directive = "ABORT_AND_DIVERT_FOB"
            operational_advisory = f"Imminent failure hazard. Abort ISR sortie immediately. Derate power to {max_safe_throttle_pct:.0f}% and vector towards {best_divert_base or 'FOB-Alpha'}."
        else:
            mission_directive = "EMERGENCY_RECOVERY_NOW"
            operational_advisory = "Critical propulsion degradation! Execute emergency glide and divert to nearest recovery strip immediately."

        return {
            "mission_directive": mission_directive,
            "max_safe_throttle_pct": round(max_safe_throttle_pct, 0),
            "derate_reasons": derate_reasons if derate_reasons else ["All parameters within standard operating limits"],
            "total_reachable_range_nm": round(total_reachable_range_nm, 1),
            "powered_range_nm": round(powered_range_nm, 1),
            "deadstick_glide_range_nm": round(glide_range_nm, 1),
            "endurance_remaining_hours": round(powered_endurance_hrs, 2),
            "operational_advisory": operational_advisory,
            "recovery_bases": base_status
        }

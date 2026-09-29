"""
Multi-Mission Engine Flight Simulator
Simulates MALE UAV flight operations across diverse mission profiles:
- High Altitude Reconnaissance (up to 25,000 ft)
- Hot Desert Operations (+46°C ambient)
- Rapid Throttle & Tactical Maneuvering
- Maritime Long Endurance Patrol
- Controlled Fault Diagnostic Evaluations
"""

import json
import time
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..physics.atmosphere import AtmosphereModel
from ..physics.thermodynamic_cycle import ThermodynamicEngineModel
from ..physics.turbocharger import TurbochargerModel
from ..physics.thermal_network import ThermalNetworkModel
from ..physics.vibration_model import VibrationSignatureModel
from ..core.state import TelemetryFrame
from .fault_injector import FaultInjector

class MissionSimulator:
    def __init__(self, profiles_path: str = None, fault_injector: FaultInjector = None):
        self.atmos_model = AtmosphereModel()
        self.thermo_model = ThermodynamicEngineModel()
        self.turbo_model = TurbochargerModel()
        self.thermal_model = ThermalNetworkModel()
        self.vib_model = VibrationSignatureModel()
        self.fault_injector = fault_injector or FaultInjector()

        # Load mission profiles
        if profiles_path is None:
            profiles_path = str(Path(__file__).parent.parent.parent / "config" / "mission_profiles.json")
        with open(profiles_path, "r") as f:
            self.profiles = json.load(f)

        self.current_profile_key = "high_altitude_recon"
        self.mission_time_s = 0.0
        self.current_phase_index = 0
        self.phase_time_s = 0.0
        
        # Engine dynamics state
        self.current_rpm = 1500.0
        self.current_throttle_pct = 25.0
        self.current_altitude_ft = 500.0
        self.current_airspeed_kts = 0.0
        self.current_ambient_temp_c = 15.0

    def set_profile(self, profile_key: str):
        if profile_key in self.profiles:
            self.current_profile_key = profile_key
            self.mission_time_s = 0.0
            self.current_phase_index = 0
            self.phase_time_s = 0.0
            # Reset thermal states
            self.thermal_model = ThermalNetworkModel()
            self.fault_injector.clear_all()

    def get_available_profiles(self) -> List[Dict[str, Any]]:
        return [
            {"key": k, "name": v.get("name", k), "description": v.get("description", "")}
            for k, v in self.profiles.items()
        ]

    def step(self, dt_s: float = 0.1) -> TelemetryFrame:
        """
        Advance simulation by dt_s seconds and generate a realistic TelemetryFrame.
        """
        profile = self.profiles[self.current_profile_key]
        phases = profile["phases"]
        if self.current_phase_index >= len(phases):
            # Loop or hold in final phase
            current_phase = phases[-1]
        else:
            current_phase = phases[self.current_phase_index]

        self.mission_time_s += dt_s
        self.phase_time_s += dt_s

        # Check phase transition
        if self.phase_time_s >= current_phase["duration_s"] and self.current_phase_index < len(phases) - 1:
            self.current_phase_index += 1
            self.phase_time_s = 0.0
            current_phase = phases[self.current_phase_index]
            # Check if this phase specifies an automatic fault injection
            if "inject_fault" in current_phase:
                self.fault_injector.inject_fault(
                    fault_type=current_phase["inject_fault"],
                    severity=current_phase.get("fault_severity", 0.5),
                    target_cylinder=current_phase.get("target_cylinder", 2)
                )

        # Smoothly slew environmental and flight targets
        target_alt = current_phase["altitude_ft"]
        target_temp = current_phase["ambient_temp_c"]
        target_throttle = current_phase["throttle_pct"]
        target_airspeed = current_phase["airspeed_kts"]

        alt_alpha = min(1.0, dt_s / 8.0)
        temp_alpha = min(1.0, dt_s / 6.0)
        throt_alpha = min(1.0, dt_s / 0.8) # Faster throttle movement
        speed_alpha = min(1.0, dt_s / 4.0)

        self.current_altitude_ft += alt_alpha * (target_alt - self.current_altitude_ft)
        self.current_ambient_temp_c += temp_alpha * (target_temp - self.current_ambient_temp_c)
        self.current_throttle_pct += throt_alpha * (target_throttle - self.current_throttle_pct)
        self.current_airspeed_kts += speed_alpha * (target_airspeed - self.current_airspeed_kts)

        # RPM responds to throttle with engine inertia and propeller governor load
        # Idle = 1400, Takeoff = 5800 RPM
        target_rpm = 1400.0 + (self.current_throttle_pct / 100.0) * (5800.0 - 1400.0)
        rpm_alpha = min(1.0, dt_s / 0.6)
        self.current_rpm += rpm_alpha * (target_rpm - self.current_rpm)

        # ---------------------------------------------
        # Evaluate Active Fault Injections
        # ---------------------------------------------
        faults = self.fault_injector.get_state()
        
        cyl_fuel_multipliers = [1.0, 1.0, 1.0, 1.0]
        misfire_mask = [False, False, False, False]
        coolant_restriction = 0.0
        oil_degradation = 0.0
        oil_leak = 0.0
        bearing_wear = 0.0
        knock_severity = 0.0
        unbalance_severity = 0.0
        sensor_drift_bias = {"channel": None, "offset": 0.0}

        # 1. Misfire
        if "misfire" in faults:
            f = faults["misfire"]
            target_cyl = f["target_cylinder"] - 1
            misfire_mask[target_cyl] = True
            # Engine RPM experiences cyclic hesitation
            self.current_rpm -= 180.0 * f["severity"]

        # 2. Injector Abnormality
        if "injector_abnormal" in faults:
            f = faults["injector_abnormal"]
            target_cyl = f["target_cylinder"] - 1
            # Lean bias: delivers 40-70% less fuel
            cyl_fuel_multipliers[target_cyl] = max(0.2, 1.0 - 0.65 * f["severity"])

        # 3. Cooling Degradation
        if "cooling_degradation" in faults:
            f = faults["cooling_degradation"]
            coolant_restriction = f["severity"]

        # 4. Lubrication Issue
        if "lubrication_issue" in faults:
            f = faults["lubrication_issue"]
            oil_leak = f["severity"]
            oil_degradation = f["severity"] * 0.7
            bearing_wear = f["severity"] * 0.5

        # 5. Sensor Drift
        if "sensor_drift" in faults:
            f = faults["sensor_drift"]
            sensor_drift_bias["channel"] = f.get("params", {}).get("channel", "cht_2_c")
            sensor_drift_bias["offset"] = 28.0 * f["severity"]

        # 6. Combustion Instability / Knock
        if "combustion_instability" in faults:
            f = faults["combustion_instability"]
            knock_severity = f["severity"]

        # 7. Abnormal Vibration
        if "abnormal_vibration" in faults:
            f = faults["abnormal_vibration"]
            unbalance_severity = f["severity"] * 0.8
            bearing_wear = max(bearing_wear, f["severity"] * 0.6)

        # 8. Overheating Trend
        if "overheating_trend" in faults:
            f = faults["overheating_trend"]
            coolant_restriction = max(coolant_restriction, 0.45 * f["severity"])

        # ---------------------------------------------
        # Physics Calculations
        # ---------------------------------------------
        # Atmosphere
        atmos = self.atmos_model.calculate(self.current_altitude_ft, self.current_ambient_temp_c)
        ambient_p_pa = atmos["pressure_pa"]
        ambient_t_k = atmos["temperature_k"]

        # Turbocharger
        turbo = self.turbo_model.compute(
            ambient_pressure_pa=ambient_p_pa,
            ambient_temp_k=ambient_t_k,
            throttle_pct=self.current_throttle_pct,
            rpm=self.current_rpm,
            airspeed_kts=self.current_airspeed_kts
        )
        map_pa = turbo["map_pa"]
        map_inhg = turbo["map_inhg"]

        # Thermodynamic cycle
        thermo = self.thermo_model.calculate_state(
            rpm=self.current_rpm,
            manifold_pressure_pa=map_pa,
            manifold_temp_k=turbo["manifold_temp_k"],
            air_fuel_ratio=14.2,
            combustion_efficiency=0.98 if not any(misfire_mask) else 0.75,
            mechanical_wear_factor=1.0 + bearing_wear * 0.3
        )

        # Thermal network
        thermal = self.thermal_model.step(
            dt_s=dt_s,
            rpm=self.current_rpm,
            power_indicated_kw=thermo["power_indicated_kw"],
            power_brake_kw=thermo["power_brake_kw"],
            mass_fuelflow_kg_s=thermo["mass_fuelflow_kg_s"],
            ambient_temp_c=self.current_ambient_temp_c,
            airspeed_kts=self.current_airspeed_kts,
            air_fuel_ratio=14.2,
            cylinder_fuel_multipliers=cyl_fuel_multipliers,
            misfire_mask=misfire_mask,
            coolant_restriction_factor=coolant_restriction,
            oil_degradation_factor=oil_degradation,
            oil_leak_factor=oil_leak
        )

        # Vibration model
        vib = self.vib_model.compute(
            rpm=self.current_rpm,
            power_brake_kw=thermo["power_brake_kw"],
            bmep_bar=thermo["bmep_bar"],
            misfire_active=any(misfire_mask),
            bearing_wear_severity=bearing_wear,
            propeller_unbalance_severity=unbalance_severity,
            combustion_knock_severity=knock_severity
        )

        # Add realistic sensor measurement noise
        noise_cht = [random.gauss(0.0, 0.4) for _ in range(4)]
        noise_egt = [random.gauss(0.0, 1.8) for _ in range(4)]
        noise_oil_p = random.gauss(0.0, 0.02)
        noise_oil_t = random.gauss(0.0, 0.3)
        noise_ff = random.gauss(0.0, 0.15)
        noise_vib = random.gauss(0.0, 0.03)

        c1 = thermal["cht_1_c"] + noise_cht[0]
        c2 = thermal["cht_2_c"] + noise_cht[1]
        c3 = thermal["cht_3_c"] + noise_cht[2]
        c4 = thermal["cht_4_c"] + noise_cht[3]

        # Apply sensor drift fault if active
        if sensor_drift_bias["channel"] == "cht_1_c": c1 += sensor_drift_bias["offset"]
        elif sensor_drift_bias["channel"] == "cht_2_c": c2 += sensor_drift_bias["offset"]
        elif sensor_drift_bias["channel"] == "cht_3_c": c3 += sensor_drift_bias["offset"]
        elif sensor_drift_bias["channel"] == "cht_4_c": c4 += sensor_drift_bias["offset"]

        e1 = thermal["egt_1_c"] + noise_egt[0]
        e2 = thermal["egt_2_c"] + noise_egt[1]
        e3 = thermal["egt_3_c"] + noise_egt[2]
        e4 = thermal["egt_4_c"] + noise_egt[3]

        all_chts = [c1, c2, c3, c4]
        all_egts = [e1, e2, e3, e4]

        # Electrical bus: 28V nominal with slight alternator ripple
        bus_voltage = 28.1 + random.gauss(0.0, 0.08)
        alt_current = 22.0 + (thermo["power_brake_kw"] / 100.0) * 8.0 + random.gauss(0.0, 0.5)

        return TelemetryFrame(
            timestamp_s=time.time(),
            mission_time_s=round(self.mission_time_s, 2),
            mission_phase=current_phase["name"],
            altitude_ft=round(self.current_altitude_ft, 1),
            ambient_temp_c=round(self.current_ambient_temp_c, 1),
            ambient_pressure_pa=round(ambient_p_pa, 1),
            airspeed_kts=round(self.current_airspeed_kts, 1),
            throttle_pct=round(self.current_throttle_pct, 1),
            rpm=round(self.current_rpm + random.gauss(0.0, 3.5), 1),
            map_inhg=round(map_inhg + random.gauss(0.0, 0.05), 2),
            map_pa=round(map_pa, 1),
            cht_1_c=round(c1, 1),
            cht_2_c=round(c2, 1),
            cht_3_c=round(c3, 1),
            cht_4_c=round(c4, 1),
            cht_avg_c=round(sum(all_chts) / 4.0, 1),
            cht_spread_c=round(max(all_chts) - min(all_chts), 1),
            egt_1_c=round(e1, 1),
            egt_2_c=round(e2, 1),
            egt_3_c=round(e3, 1),
            egt_4_c=round(e4, 1),
            egt_avg_c=round(sum(all_egts) / 4.0, 1),
            egt_spread_c=round(max(all_egts) - min(all_egts), 1),
            coolant_temp_c=round(thermal["coolant_temp_c"] + random.gauss(0.0, 0.2), 1),
            oil_pressure_bar=round(max(0.4, thermal["oil_pressure_bar"] + noise_oil_p), 2),
            oil_temp_c=round(thermal["oil_temp_c"] + noise_oil_t, 1),
            fuel_flow_lph=round(max(1.0, thermo["fuel_flow_lph"] + noise_ff), 2),
            fuel_pressure_bar=round(3.0 + random.gauss(0.0, 0.03), 2),
            air_fuel_ratio=14.2,
            injection_timing_deg_btdc=round(24.0 + (self.current_rpm / 5000.0) * 4.0, 1),
            injection_pulse_width_ms=round(3.5 + (self.current_throttle_pct / 100.0) * 4.2, 2),
            ignition_timing_deg_btdc=round(26.0 - knock_severity * 4.0, 1),
            wastegate_duty_pct=round(turbo["wastegate_duty_pct"], 1),
            compressor_ratio=round(turbo["compressor_pressure_ratio"], 2),
            intercooler_exit_temp_c=round(turbo["manifold_temp_c"], 1),
            vibration_rms_g=round(max(0.2, vib["vibration_rms_g"] + noise_vib), 3),
            vibration_peak_g=round(max(0.5, vib["vibration_peak_g"]), 2),
            vibration_x_rms_g=round(vib["vibration_x_rms_g"], 3),
            vibration_y_rms_g=round(vib["vibration_y_rms_g"], 3),
            vibration_z_rms_g=round(vib["vibration_z_rms_g"], 3),
            crest_factor=round(vib["crest_factor"], 2),
            spectral_bins=vib["spectral_bins"],
            bus_voltage_v=round(bus_voltage, 2),
            alternator_current_a=round(alt_current, 1),
            battery_current_a=round(1.2 + random.gauss(0.0, 0.1), 1),
            voltage_ripple_mv=round(85.0 + knock_severity * 20.0, 1)
        )

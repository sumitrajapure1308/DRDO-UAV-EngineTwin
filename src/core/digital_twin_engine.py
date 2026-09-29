"""
Digital Twin Core Engine and State Synchronizer
Maintains a virtual replica of the aero piston engine driven by physics-based models,
computes dynamic model-telemetry residuals, and estimates unmeasured internal engine states.
"""

import math
import numpy as np
from typing import Dict, Any, Tuple
from ..physics.atmosphere import AtmosphereModel
from ..physics.thermodynamic_cycle import ThermodynamicEngineModel
from ..physics.turbocharger import TurbochargerModel
from ..physics.thermal_network import ThermalNetworkModel
from ..physics.vibration_model import VibrationSignatureModel
from .state import TelemetryFrame, DigitalTwinEstimate

class DigitalTwinSynchronizer:
    """
    Virtual Engine Model that continuously mirrors the physical aero piston engine.
    Computes expected nominal states using thermodynamic, aerodynamic, and kinematic models,
    and isolates physical discrepancies (residuals) for downstream AI diagnostics.
    """

    def __init__(self):
        self.atmos_model = AtmosphereModel()
        self.thermo_model = ThermodynamicEngineModel()
        self.turbo_model = TurbochargerModel()
        self.thermal_model = ThermalNetworkModel()
        self.vib_model = VibrationSignatureModel()
        
        self.last_timestamp_s = None
        
        # State tracking filters for smooth state observation
        self.filter_alpha = 0.25

    def synchronize(self, telemetry: TelemetryFrame) -> DigitalTwinEstimate:
        """
        Synchronize virtual engine model with the incoming telemetry frame.
        """
        # Synchronize using telemetry mission time if available, otherwise timestamp
        now = telemetry.mission_time_s if telemetry.mission_time_s > 0 else telemetry.timestamp_s
        if self.last_timestamp_s is None:
            dt_s = 0.1
        else:
            dt_s = max(0.001, min(2.0, now - self.last_timestamp_s))
        self.last_timestamp_s = now

        # 1. Physics: Atmospheric conditions at current altitude
        atmos = self.atmos_model.calculate(telemetry.altitude_ft, telemetry.ambient_temp_c)
        ambient_p_pa = atmos["pressure_pa"]
        ambient_t_k = atmos["temperature_k"]

        # 2. Physics: Turbocharger and intake charge state
        turbo = self.turbo_model.compute(
            ambient_pressure_pa=ambient_p_pa,
            ambient_temp_k=ambient_t_k,
            throttle_pct=telemetry.throttle_pct,
            rpm=telemetry.rpm,
            airspeed_kts=telemetry.airspeed_kts
        )
        twin_map_pa = turbo["map_pa"]
        twin_map_inhg = turbo["map_inhg"]
        manifold_temp_k = turbo["manifold_temp_k"]

        # 3. Physics: Thermodynamic cycle (combustion, power, torque, BSFC, fuel flow)
        # Digital twin assumes nominal air-fuel ratio 14.2:1 and nominal mechanical efficiency
        thermo = self.thermo_model.calculate_state(
            rpm=telemetry.rpm,
            manifold_pressure_pa=twin_map_pa,
            manifold_temp_k=manifold_temp_k,
            air_fuel_ratio=14.2,
            combustion_efficiency=0.98,
            mechanical_wear_factor=1.0
        )
        twin_fuel_flow_lph = thermo["fuel_flow_lph"]
        twin_power_kw = thermo["power_brake_kw"]
        twin_power_hp = thermo["power_brake_hp"]
        twin_bmep_bar = thermo["bmep_bar"]
        twin_bsfc = thermo["bsfc_g_kwh"]
        twin_eta_th = thermo["eta_brake_thermal"]

        # 4. Physics: Thermal network (nominal 4-cylinder CHT, EGT, Coolant, Oil P/T)
        thermal = self.thermal_model.step(
            dt_s=dt_s,
            rpm=telemetry.rpm,
            power_indicated_kw=thermo["power_indicated_kw"],
            power_brake_kw=twin_power_kw,
            mass_fuelflow_kg_s=thermo["mass_fuelflow_kg_s"],
            ambient_temp_c=telemetry.ambient_temp_c,
            airspeed_kts=telemetry.airspeed_kts,
            air_fuel_ratio=14.2,
            cylinder_fuel_multipliers=[1.0, 1.0, 1.0, 1.0],
            misfire_mask=[False, False, False, False],
            coolant_restriction_factor=0.0,
            oil_degradation_factor=0.0,
            oil_leak_factor=0.0
        )

        # 5. Physics: Vibration kinematics
        vib = self.vib_model.compute(
            rpm=telemetry.rpm,
            power_brake_kw=twin_power_kw,
            bmep_bar=twin_bmep_bar,
            misfire_active=False,
            bearing_wear_severity=0.0,
            propeller_unbalance_severity=0.0,
            combustion_knock_severity=0.0
        )
        twin_vib_rms_g = vib["vibration_rms_g"]

        # 6. Calculate Physics Residuals (Measured - Nominal Twin)
        # Residuals represent unmodeled degradations, faults, leaks, or drifts
        res_map = telemetry.map_pa - twin_map_pa
        res_cht_1 = telemetry.cht_1_c - thermal["cht_1_c"]
        res_cht_2 = telemetry.cht_2_c - thermal["cht_2_c"]
        res_cht_3 = telemetry.cht_3_c - thermal["cht_3_c"]
        res_cht_4 = telemetry.cht_4_c - thermal["cht_4_c"]
        
        res_egt_1 = telemetry.egt_1_c - thermal["egt_1_c"]
        res_egt_2 = telemetry.egt_2_c - thermal["egt_2_c"]
        res_egt_3 = telemetry.egt_3_c - thermal["egt_3_c"]
        res_egt_4 = telemetry.egt_4_c - thermal["egt_4_c"]
        
        res_ff = telemetry.fuel_flow_lph - twin_fuel_flow_lph
        res_oil_p = telemetry.oil_pressure_bar - thermal["oil_pressure_bar"]
        res_oil_t = telemetry.oil_temp_c - thermal["oil_temp_c"]
        res_vib = telemetry.vibration_rms_g - twin_vib_rms_g

        # Normalized Composite Residual Euclidean Norm
        # Scale each residual by typical operational standard deviations
        norm_map = res_map / 3000.0
        norm_cht = max([abs(res_cht_1), abs(res_cht_2), abs(res_cht_3), abs(res_cht_4)]) / 8.0
        norm_egt = max([abs(res_egt_1), abs(res_egt_2), abs(res_egt_3), abs(res_egt_4)]) / 35.0
        norm_ff = res_ff / 2.0
        norm_oil_p = res_oil_p / 0.5
        norm_oil_t = res_oil_t / 5.0
        norm_vib = res_vib / 0.35

        composite_norm = math.sqrt(
            norm_map**2 + norm_cht**2 + norm_egt**2 + norm_ff**2 +
            norm_oil_p**2 + norm_oil_t**2 + norm_vib**2
        )

        return DigitalTwinEstimate(
            timestamp_s=now,
            twin_map_pa=round(twin_map_pa, 1),
            twin_map_inhg=round(twin_map_inhg, 2),
            twin_power_brake_kw=round(twin_power_kw, 2),
            twin_power_brake_hp=round(twin_power_hp, 1),
            twin_bmep_bar=round(twin_bmep_bar, 2),
            twin_bsfc_g_kwh=round(twin_bsfc, 1),
            twin_eta_thermal=round(twin_eta_th, 4),
            twin_fuel_flow_lph=round(twin_fuel_flow_lph, 2),
            twin_cht_1_c=round(thermal["cht_1_c"], 1),
            twin_cht_2_c=round(thermal["cht_2_c"], 1),
            twin_cht_3_c=round(thermal["cht_3_c"], 1),
            twin_cht_4_c=round(thermal["cht_4_c"], 1),
            twin_egt_1_c=round(thermal["egt_1_c"], 1),
            twin_egt_2_c=round(thermal["egt_2_c"], 1),
            twin_egt_3_c=round(thermal["egt_3_c"], 1),
            twin_egt_4_c=round(thermal["egt_4_c"], 1),
            twin_coolant_temp_c=round(thermal["coolant_temp_c"], 1),
            twin_oil_temp_c=round(thermal["oil_temp_c"], 1),
            twin_oil_pressure_bar=round(thermal["oil_pressure_bar"], 2),
            twin_vibration_rms_g=round(twin_vib_rms_g, 3),
            residual_map_pa=round(res_map, 1),
            residual_cht_1_c=round(res_cht_1, 2),
            residual_cht_2_c=round(res_cht_2, 2),
            residual_cht_3_c=round(res_cht_3, 2),
            residual_cht_4_c=round(res_cht_4, 2),
            residual_egt_1_c=round(res_egt_1, 2),
            residual_egt_2_c=round(res_egt_2, 2),
            residual_egt_3_c=round(res_egt_3, 2),
            residual_egt_4_c=round(res_egt_4, 2),
            residual_fuel_flow_lph=round(res_ff, 2),
            residual_oil_pressure_bar=round(res_oil_p, 3),
            residual_oil_temp_c=round(res_oil_t, 2),
            residual_vibration_rms_g=round(res_vib, 3),
            composite_residual_norm=round(composite_norm, 3)
        )

"""
Health Monitoring System and Subsystem Health Index Synthesizer
Assesses RPM, CHT (1..4), EGT (1..4), Oil P/T, Fuel Flow, Vibration Harmonics,
Electrical Bus, and Injection Parameters to compute individual and composite health scores.
"""

import math
from typing import Dict, Any, List
from ..core.state import TelemetryFrame, DigitalTwinEstimate, SubsystemHealth, CompositeEngineHealth

class HealthMonitoringSystem:
    def __init__(self):
        # Weights for Composite Engine Health Index (sum = 1.0)
        self.weights = {
            "thermal_cht": 0.16,
            "exhaust_gas_path": 0.14,
            "fuel_injection": 0.14,
            "lubrication_system": 0.18,
            "turbo_charge": 0.10,
            "cooling_system": 0.10,
            "combustion_stability": 0.12,
            "electrical_bus": 0.06
        }
        self.smoothed_health_index = 1.0
        self.smoothing_factor = 0.15

    def evaluate(self, telem: TelemetryFrame, twin: DigitalTwinEstimate) -> CompositeEngineHealth:
        """
        Evaluate engine subsystems and synthesize Composite Engine Health.
        """
        # 1. Thermal CHT Health (0 - 100)
        # Redline is 145°C, Caution is 135°C, Ideal 90-110°C
        max_cht = max(telem.cht_1_c, telem.cht_2_c, telem.cht_3_c, telem.cht_4_c)
        if max_cht < 115.0:
            h_cht_abs = 100.0
        elif max_cht < 135.0:
            h_cht_abs = 100.0 - ((max_cht - 115.0) / 20.0) * 35.0
        elif max_cht < 145.0:
            h_cht_abs = 65.0 - ((max_cht - 135.0) / 10.0) * 45.0
        else:
            h_cht_abs = max(0.0, 20.0 - (max_cht - 145.0) * 5.0)

        # CHT Spread penalty (spread > 18°C indicates localized flow obstruction)
        spread_penalty = max(0.0, (telem.cht_spread_c - 12.0) * 2.5)
        h_thermal = max(0.0, min(100.0, h_cht_abs - spread_penalty))

        # 2. Exhaust Gas Path Health (0 - 100)
        # Normal 720-820°C, Caution 840°C, Redline 880°C
        max_egt = max(telem.egt_1_c, telem.egt_2_c, telem.egt_3_c, telem.egt_4_c)
        min_egt = min(telem.egt_1_c, telem.egt_2_c, telem.egt_3_c, telem.egt_4_c)
        if max_egt < 825.0:
            h_egt_abs = 100.0
        elif max_egt < 860.0:
            h_egt_abs = 100.0 - ((max_egt - 825.0) / 35.0) * 40.0
        else:
            h_egt_abs = max(0.0, 60.0 - ((max_egt - 860.0) / 20.0) * 50.0)

        # EGT spread penalty (spread > 45°C indicates injector or valve sealing issue)
        egt_spread_pen = max(0.0, (telem.egt_spread_c - 35.0) * 1.2)
        # Deep misfire check (EGT below 400°C while engine is operating above idle)
        if min_egt < 450.0 and telem.rpm > 2500.0:
            h_egt_abs = min(h_egt_abs, 25.0)
        h_exhaust = max(0.0, min(100.0, h_egt_abs - egt_spread_pen))

        # 3. Fuel Injection Health (0 - 100)
        # Fuel pressure nominal 2.8 - 3.2 bar
        fp_error = abs(telem.fuel_pressure_bar - 3.0)
        h_fp = 100.0 - (fp_error / 0.5) * 50.0 if fp_error > 0.15 else 100.0
        # Fuel flow residual penalty (measured vs twin expected)
        ff_residual_pct = abs(twin.residual_fuel_flow_lph) / max(1.0, twin.twin_fuel_flow_lph)
        h_ff = 100.0 - max(0.0, (ff_residual_pct - 0.08) * 300.0)
        h_fuel = max(0.0, min(100.0, 0.4 * h_fp + 0.6 * h_ff))

        # 4. Lubrication System Health (0 - 100)
        # Minimum safe pressure 1.5 bar, nominal 3.0-5.0 bar, redline low 0.8 bar
        op = telem.oil_pressure_bar
        ot = telem.oil_temp_c
        if op >= 3.0:
            h_op = 100.0
        elif op >= 2.0:
            h_op = 80.0 + ((op - 2.0) / 1.0) * 20.0
        elif op >= 1.2:
            h_op = 40.0 + ((op - 1.2) / 0.8) * 40.0
        else:
            h_op = max(0.0, (op / 1.2) * 40.0)

        # Oil temperature: nominal 80-105°C, caution 118°C, max 130°C
        if ot <= 105.0:
            h_ot = 100.0
        elif ot <= 120.0:
            h_ot = 100.0 - ((ot - 105.0) / 15.0) * 35.0
        else:
            h_ot = max(0.0, 65.0 - ((ot - 120.0) / 10.0) * 55.0)
        h_lube = max(0.0, min(100.0, 0.65 * h_op + 0.35 * h_ot))

        # 5. Turbocharger & Boost Health (0 - 100)
        map_err_pct = abs(twin.residual_map_pa) / max(20000.0, twin.twin_map_pa)
        h_map = 100.0 - max(0.0, (map_err_pct - 0.06) * 400.0)
        # Wastegate saturation check (if 100% duty cycle but MAP not met)
        if telem.wastegate_duty_pct > 96.0 and twin.residual_map_pa < -5000.0:
            h_map = min(h_map, 55.0)
        h_turbo = max(0.0, min(100.0, h_map))

        # 6. Cooling System Health (0 - 100)
        ct = telem.coolant_temp_c
        if ct <= 98.0:
            h_cool = 100.0
        elif ct <= 112.0:
            h_cool = 100.0 - ((ct - 98.0) / 14.0) * 40.0
        else:
            h_cool = max(0.0, 60.0 - ((ct - 112.0) / 15.0) * 55.0)

        # 7. Combustion Stability & Vibration Health (0 - 100)
        # RMS vibration nominal < 1.8g, caution 2.5g, warning 3.5g, critical > 4.5g
        vib_rms = telem.vibration_rms_g
        if vib_rms <= 1.8:
            h_vib = 100.0
        elif vib_rms <= 2.8:
            h_vib = 100.0 - ((vib_rms - 1.8) / 1.0) * 35.0
        elif vib_rms <= 4.0:
            h_vib = 65.0 - ((vib_rms - 2.8) / 1.2) * 45.0
        else:
            h_vib = max(0.0, 20.0 - (vib_rms - 4.0) * 20.0)

        # Crest factor penalty (shocks from detonation or misfire)
        crest_pen = max(0.0, (telem.crest_factor - 3.0) * 15.0)
        h_comb = max(0.0, min(100.0, h_vib - crest_pen))

        # 8. Electrical Bus Health (0 - 100)
        volt = telem.bus_voltage_v
        if 27.2 <= volt <= 28.8:
            h_volt = 100.0
        elif 25.5 <= volt < 27.2:
            h_volt = 75.0 + ((volt - 25.5) / 1.7) * 25.0
        elif 28.8 < volt <= 30.0:
            h_volt = 100.0 - ((volt - 28.8) / 1.2) * 30.0
        else:
            h_volt = max(0.0, 45.0 - abs(volt - 28.0) * 10.0)
        # AC ripple penalty (diode failure)
        ripple_pen = max(0.0, (telem.voltage_ripple_mv - 150.0) * 0.15)
        h_elec = max(0.0, min(100.0, h_volt - ripple_pen))

        sub_scores = SubsystemHealth(
            thermal_cht=round(h_thermal, 1),
            exhaust_gas_path=round(h_exhaust, 1),
            fuel_injection=round(h_fuel, 1),
            lubrication_system=round(h_lube, 1),
            turbo_charge=round(h_turbo, 1),
            cooling_system=round(h_cool, 1),
            combustion_stability=round(h_comb, 1),
            electrical_bus=round(h_elec, 1)
        )

        # Weighted aggregate health index (0.0 to 1.0)
        raw_ehi = (
            sub_scores.thermal_cht * self.weights["thermal_cht"] +
            sub_scores.exhaust_gas_path * self.weights["exhaust_gas_path"] +
            sub_scores.fuel_injection * self.weights["fuel_injection"] +
            sub_scores.lubrication_system * self.weights["lubrication_system"] +
            sub_scores.turbo_charge * self.weights["turbo_charge"] +
            sub_scores.cooling_system * self.weights["cooling_system"] +
            sub_scores.combustion_stability * self.weights["combustion_stability"] +
            sub_scores.electrical_bus * self.weights["electrical_bus"]
        ) / 100.0

        # Exponential smoothing to filter transient high-frequency blips
        self.smoothed_health_index = (
            (1.0 - self.smoothing_factor) * self.smoothed_health_index +
            self.smoothing_factor * raw_ehi
        )

        # Physical safety margins
        thermal_margin_c = max(0.0, 145.0 - max_cht)
        oil_p_margin_bar = max(0.0, telem.oil_pressure_bar - 1.2)

        # Mission reliability probability estimation (%)
        # Based on weakest subsystem bottleneck
        weakest_subsystem = min([
            h_thermal, h_exhaust, h_fuel, h_lube,
            h_turbo, h_cool, h_comb, h_elec
        ])
        mission_reliability_pct = max(5.0, min(99.9, (self.smoothed_health_index ** 1.5) * (weakest_subsystem / 100.0) ** 0.5 * 100.0))

        return CompositeEngineHealth(
            overall_health_index=round(self.smoothed_health_index, 4),
            subsystem_scores=sub_scores,
            degradation_rate_per_hr=round((1.0 - self.smoothed_health_index) * 0.05, 5),
            mission_reliability_pct=round(mission_reliability_pct, 1),
            thermal_margin_c=round(thermal_margin_c, 1),
            oil_pressure_margin_bar=round(oil_p_margin_bar, 2)
        )

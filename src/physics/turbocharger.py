"""
Turbocharger and Charge Air Intercooler Physics Model
Models compressor pressure ratio, turbine expansion, wastegate actuator dynamics,
intercooler heat dissipation, and turbo lag for MALE UAV flight envelopes.
"""

import math
from typing import Dict, Any

class TurbochargerModel:
    def __init__(self):
        self.gamma_air = 1.4
        self.gamma_exh = 1.33
        self.cp_air = 1005.0       # J/(kg·K)
        self.cp_exh = 1150.0       # J/(kg·K)
        self.eta_compressor_nom = 0.76
        self.eta_turbine_nom = 0.74
        self.intercooler_eff_nom = 0.78
        self.max_boost_ratio = 2.45
        self.turbo_inertia_time_constant = 0.45  # seconds for spool-up / spool-down

    def compute(
        self,
        ambient_pressure_pa: float,
        ambient_temp_k: float,
        throttle_pct: float,
        rpm: float,
        airspeed_kts: float = 85.0,
        wastegate_bias: float = 0.0,       # 0.0 nominal, >0 = stuck open, <0 = overboost
        intercooler_fouling: float = 0.0   # 0.0 clean, 1.0 completely clogged
    ) -> Dict[str, float]:
        """
        Calculates manifold pressure, compressor delivery temp, and intercooler exit temp.
        """
        p_amb = max(20000.0, float(ambient_pressure_pa))
        t_amb = max(210.0, float(ambient_temp_k))
        throttle = max(0.0, min(100.0, float(throttle_pct))) / 100.0
        rpm = max(1000.0, float(rpm))

        # Target Manifold Absolute Pressure (MAP in Pa) based on throttle demand
        # Idle (throttle=0): MAP ~ 0.45 * P_amb (deep throttle manifold vacuum)
        # Cruise (throttle=0.7): MAP ~ 1.05 * P0_sea_level (~ 31 inHg)
        # Full Takeoff (throttle=1.0): MAP ~ 1.35 * P0_sea_level (~ 39 inHg = 137 kPa)
        p0_sea = 101325.0
        min_map = 0.45 * p_amb
        max_map_target = 1.38 * p0_sea  # ~ 139 kPa (41 inHg)

        # Non-linear throttle plate curve
        effective_demand = (throttle ** 1.3)
        target_map = min_map + effective_demand * (max_map_target - min_map)

        # Wastegate control loop
        # The ECU modulates wastegate duty cycle to hold target MAP
        # At sea level cruise, wastegate is open ~50-60%.
        # At high altitude, wastegate closes towards 100% duty cycle.
        needed_boost_ratio = target_map / p_amb
        max_achievable_boost = min(self.max_boost_ratio, 1.0 + (rpm / 5500.0) * (self.max_boost_ratio - 1.0))
        
        # Apply wastegate bias (fault injection: wastegate leak or stuck)
        effective_max_boost = max(1.0, max_achievable_boost * (1.0 - wastegate_bias * 0.45))
        
        actual_boost_ratio = min(effective_max_boost, max(0.45, needed_boost_ratio))
        map_pa = p_amb * actual_boost_ratio
        
        # Wastegate duty cycle percentage (0% = fully open / minimum boost, 100% = fully closed / maximum boost)
        wastegate_duty_pct = min(100.0, max(0.0, ((actual_boost_ratio - 1.0) / max(0.01, self.max_boost_ratio - 1.0)) * 100.0))

        # Compressor discharge temperature (isentropic formula)
        eta_c = max(0.60, self.eta_compressor_nom * (1.0 - 0.05 * abs(actual_boost_ratio - 1.8)))
        if actual_boost_ratio > 1.0:
            exp_term = (actual_boost_ratio) ** ((self.gamma_air - 1.0) / self.gamma_air) - 1.0
            t_comp_out_k = t_amb * (1.0 + exp_term / eta_c)
        else:
            t_comp_out_k = t_amb

        # Intercooler effectiveness depends on ram airspeed
        ram_speed_factor = min(1.0, max(0.3, airspeed_kts / 90.0))
        effective_ic_eff = self.intercooler_eff_nom * ram_speed_factor * (1.0 - intercooler_fouling * 0.6)
        
        # Manifold intake temperature after intercooler
        t_manifold_k = t_comp_out_k - effective_ic_eff * max(0.0, t_comp_out_k - t_amb)

        return {
            "map_pa": float(map_pa),
            "map_inhg": float(map_pa / 3386.389),
            "map_bar": float(map_pa / 100000.0),
            "compressor_pressure_ratio": float(actual_boost_ratio),
            "compressor_out_temp_k": float(t_comp_out_k),
            "compressor_out_temp_c": float(t_comp_out_k - 273.15),
            "manifold_temp_k": float(t_manifold_k),
            "manifold_temp_c": float(t_manifold_k - 273.15),
            "wastegate_duty_pct": float(wastegate_duty_pct),
            "intercooler_effectiveness": float(effective_ic_eff)
        }

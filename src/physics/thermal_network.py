"""
Multi-Node Thermal Network and Heat Rejection Physics Model
Simulates individual cylinder head temperatures (CHT 1..4), exhaust gas temperatures (EGT 1..4),
engine coolant heat rejection, and oil temperature/pressure dynamics under transient aero flight conditions.
"""

import math
from typing import Dict, Any, List

class ThermalNetworkModel:
    def __init__(self):
        # Thermal capacities (J/K)
        self.c_head = 4800.0         # Thermal mass of each cylinder head
        self.c_coolant = 12000.0     # Coolant loop thermal capacity
        self.c_oil = 9500.0          # Oil sump + cooler thermal mass

        # Geometric airflow / thermal distribution factors for 4-cylinder layout
        # (Cyl 1 & 2 front facing ram air; Cyl 3 & 4 rear shielded in nacelle)
        self.cyl_airflow_bias = [1.02, 1.01, 0.98, 0.97]
        self.cyl_exhaust_runner_bias = [0.99, 1.01, 1.02, 0.98]

        # Initial state temperatures (°C)
        self.cht = [85.0, 86.0, 88.0, 87.0]
        self.egt = [740.0, 745.0, 755.0, 742.0]
        self.coolant_temp_c = 82.0
        self.oil_temp_c = 85.0
        self.oil_pressure_bar = 4.2

    def step(
        self,
        dt_s: float,
        rpm: float,
        power_indicated_kw: float,
        power_brake_kw: float,
        mass_fuelflow_kg_s: float,
        ambient_temp_c: float,
        airspeed_kts: float,
        air_fuel_ratio: float = 14.2,
        cylinder_fuel_multipliers: List[float] = None,  # Injector balance [1.0, 1.0, 1.0, 1.0]
        misfire_mask: List[bool] = None,               # [False, False, False, False]
        coolant_restriction_factor: float = 0.0,       # 0.0 nominal, 1.0 completely blocked
        oil_degradation_factor: float = 0.0,           # 0.0 fresh oil, 1.0 worn/diluted
        oil_leak_factor: float = 0.0                   # 0.0 no leak, 1.0 severe pressure loss
    ) -> Dict[str, Any]:
        """
        Advance thermal state by dt_s seconds.
        """
        if cylinder_fuel_multipliers is None:
            cylinder_fuel_multipliers = [1.0, 1.0, 1.0, 1.0]
        if misfire_mask is None:
            misfire_mask = [False, False, False, False]

        rpm = max(800.0, float(rpm))
        airspeed_kts = max(0.0, float(airspeed_kts))
        ram_speed_factor = min(1.3, max(0.2, (airspeed_kts + 15.0) / 95.0))

        # Overall heat input from combustion: approx 30% goes to coolant/heads, 35% to exhaust, 5% to oil
        fuel_energy_rate_w = max(1000.0, mass_fuelflow_kg_s * 43.5e6)
        total_thermal_rejection_w = fuel_energy_rate_w - (power_brake_kw * 1000.0)

        head_heat_share_w = total_thermal_rejection_w * 0.32
        oil_heat_share_w = total_thermal_rejection_w * 0.08
        exhaust_heat_share_w = total_thermal_rejection_w * 0.60

        # --- 1. Coolant System Dynamics ---
        # Heat dissipation to ambient via radiator
        # Coolant pump flow rate proportional to RPM
        coolant_pump_flow_ratio = (rpm / 5000.0) * (1.0 - coolant_restriction_factor * 0.85)
        radiator_ua = 420.0 * (1.0 - coolant_restriction_factor * 0.4) * ram_speed_factor  # W/K
        q_rad_out_w = radiator_ua * max(0.0, (self.coolant_temp_c - ambient_temp_c))
        
        # Heat transfer from 4 cylinder heads into coolant
        q_heads_to_coolant_w = 0.0
        for i in range(4):
            q_head_conv = 140.0 * coolant_pump_flow_ratio * (self.cht[i] - self.coolant_temp_c)
            q_heads_to_coolant_w += q_head_conv

        d_coolant = ((q_heads_to_coolant_w - q_rad_out_w) / self.c_coolant) * dt_s
        self.coolant_temp_c = max(ambient_temp_c, min(140.0, self.coolant_temp_c + d_coolant))

        # --- 2. Cylinder Head Temperatures (CHT 1..4) ---
        for i in range(4):
            if misfire_mask[i]:
                cyl_fuel_factor = 0.0
            else:
                cyl_fuel_factor = cylinder_fuel_multipliers[i]

            q_in_cyl_w = (head_heat_share_w / 4.0) * cyl_fuel_factor
            q_to_coolant = 140.0 * coolant_pump_flow_ratio * (self.cht[i] - self.coolant_temp_c)
            # Direct fin cooling by ram air
            q_ram_air = 65.0 * ram_speed_factor * self.cyl_airflow_bias[i] * (self.cht[i] - ambient_temp_c)

            d_cht = ((q_in_cyl_w - q_to_coolant - q_ram_air) / self.c_head) * dt_s
            self.cht[i] = max(ambient_temp_c, min(160.0, self.cht[i] + d_cht))

        # --- 3. Exhaust Gas Temperatures (EGT 1..4) ---
        # Baseline EGT curve as function of local AFR and power
        # Peak EGT around stoichiometric (AFR 14.7), drops with enrichment or fuel starvation
        for i in range(4):
            if misfire_mask[i]:
                # In misfire, no combustion, EGT collapses towards fresh intake/exhaust wash (~200°C)
                target_egt = 180.0 + 0.3 * (self.cht[i])
            else:
                local_fuel_mult = cylinder_fuel_multipliers[i]
                local_afr = air_fuel_ratio / max(0.1, local_fuel_mult)
                # Curve: max EGT at AFR 14.7 ~ 830°C
                afr_delta = local_afr - 14.7
                egt_afr_correction = -45.0 * (afr_delta ** 2) if afr_delta < 0 else -60.0 * afr_delta
                power_scaling = 680.0 + 130.0 * (power_brake_kw / 105.0)
                target_egt = (power_scaling + egt_afr_correction) * self.cyl_exhaust_runner_bias[i]
            
            # Fast thermocouple response (time constant ~ 1.2 s)
            alpha_egt = min(1.0, dt_s / 1.2)
            self.egt[i] = self.egt[i] + alpha_egt * (target_egt - self.egt[i])

        # --- 4. Lubrication System (Oil Temperature & Pressure) ---
        # Oil receives heat from friction + piston oil squirters
        oil_cooler_ua = 160.0 * ram_speed_factor * (1.0 - oil_degradation_factor * 0.25)
        q_oil_out_w = oil_cooler_ua * max(0.0, (self.oil_temp_c - ambient_temp_c))
        q_oil_in_w = oil_heat_share_w * (power_brake_kw / 100.0)

        d_oil_temp = ((q_oil_in_w - q_oil_out_w) / self.c_oil) * dt_s
        self.oil_temp_c = max(ambient_temp_c, min(145.0, self.oil_temp_c + d_oil_temp))

        # Oil viscosity correlation with temperature (SAE 15W-50 Aero Oil)
        # Viscosity drops exponentially with temperature
        ref_temp = self.oil_temp_c + 273.15
        viscosity_cst = 140.0 * math.exp(-0.035 * (self.oil_temp_c - 40.0))
        # Oil degradation thins oil or causes sludge
        effective_viscosity = max(6.0, viscosity_cst * (1.0 - oil_degradation_factor * 0.35))

        # Pump delivers pressure based on RPM and viscosity, capped by relief valve (5.2 bar)
        base_pressure = (rpm / 5000.0) * (2.8 + 0.04 * effective_viscosity)
        relief_capped = min(5.2, base_pressure)
        # Leaks or bearing wear reduce oil pressure directly
        leak_loss = oil_leak_factor * 2.8 + oil_degradation_factor * 0.8
        target_oil_pressure = max(0.5, relief_capped - leak_loss)

        alpha_oil_p = min(1.0, dt_s / 0.5)
        self.oil_pressure_bar = self.oil_pressure_bar + alpha_oil_p * (target_oil_pressure - self.oil_pressure_bar)

        return {
            "cht_1_c": float(self.cht[0]),
            "cht_2_c": float(self.cht[1]),
            "cht_3_c": float(self.cht[2]),
            "cht_4_c": float(self.cht[3]),
            "cht_avg_c": float(sum(self.cht) / 4.0),
            "cht_spread_c": float(max(self.cht) - min(self.cht)),
            "egt_1_c": float(self.egt[0]),
            "egt_2_c": float(self.egt[1]),
            "egt_3_c": float(self.egt[2]),
            "egt_4_c": float(self.egt[3]),
            "egt_avg_c": float(sum(self.egt) / 4.0),
            "egt_spread_c": float(max(self.egt) - min(self.egt)),
            "coolant_temp_c": float(self.coolant_temp_c),
            "oil_temp_c": float(self.oil_temp_c),
            "oil_pressure_bar": float(self.oil_pressure_bar)
        }

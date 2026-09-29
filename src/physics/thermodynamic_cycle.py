"""
4-Stroke Turbocharged Aero Piston Engine Thermodynamic Cycle Model
Implements Otto/Dual air-standard cycle with intake charge dynamics, Chen-Flynn
friction estimation, BMEP, BSFC, and brake thermal efficiency for MALE UAV operations.
"""

import math
import numpy as np
from typing import Dict, Any

class ThermodynamicEngineModel:
    def __init__(self, specs: Dict[str, Any] = None):
        self.cylinders = 4
        self.displacement_m3 = 1352.0 * 1e-6  # 1.352 L -> m^3
        self.compression_ratio = 9.0
        self.gamma = 1.33                      # Combustion products specific heat ratio
        self.lhv_fuel_j_kg = 43.5 * 1e6        # 43.5 MJ/kg AVGAS / Mogas
        self.fuel_density_kg_m3 = 720.0        # kg/m^3 (approx 0.72 kg/L)
        self.r_gas = 287.05                    # J/(kg*K)
        
        # Friction model constants (Chen-Flynn empirical correlation for aero piston engines)
        self.fmep_base = 0.35                  # bar
        self.fmep_rpm_coef = 0.18              # bar per 1000 RPM
        
        # Mechanical limits
        self.rated_power_kw = 105.0            # approx 141 HP
        self.rated_rpm = 5800.0

    def calculate_state(
        self,
        rpm: float,
        manifold_pressure_pa: float,
        manifold_temp_k: float,
        air_fuel_ratio: float = 14.2,
        combustion_efficiency: float = 0.98,
        mechanical_wear_factor: float = 1.0     # 1.0 = nominal, >1.0 = worn rings/friction
    ) -> Dict[str, float]:
        """
        Evaluate thermodynamic cycle states given instantaneous RPM and intake manifold conditions.
        """
        rpm = max(800.0, float(rpm))
        map_pa = max(20000.0, float(manifold_pressure_pa))
        t_intake = max(220.0, float(manifold_temp_k))
        
        # 1. Manifold air density (kg/m^3)
        rho_intake = map_pa / (self.r_gas * t_intake)

        # 2. Volumetric Efficiency (tuned for 4-valve pentroof aero piston engine)
        rpm_ratio = rpm / 5000.0
        eta_v_base = 0.94 * math.exp(-0.5 * ((rpm_ratio - 1.0) / 0.45) ** 2)
        # Boost correction on volumetric efficiency
        map_bar = map_pa / 100000.0
        eta_v = max(0.68, min(0.97, eta_v_base * (1.0 - 0.02 * max(0.0, map_bar - 1.0))))

        # 3. Mass Airflow Rate (kg/s)
        v_d_per_sec = (self.displacement_m3 * (rpm / 60.0)) / 2.0
        mass_airflow_kg_s = rho_intake * v_d_per_sec * eta_v

        # 4. Fuel Flow Rate (kg/s and L/h)
        afr = max(10.0, min(18.0, float(air_fuel_ratio)))
        mass_fuelflow_kg_s = mass_airflow_kg_s / afr
        fuelflow_lph = (mass_fuelflow_kg_s * 3600.0 / (self.fuel_density_kg_m3 / 1000.0))

        # 5. Indicated Thermodynamic Efficiency (Otto/Dual cycle approximation)
        eta_ideal = 1.0 - (1.0 / (self.compression_ratio ** (self.gamma - 1.0)))
        real_cycle_factor = 0.74 * combustion_efficiency
        eta_indicated = eta_ideal * real_cycle_factor

        # 6. Indicated Power (kW)
        fuel_energy_rate_kw = (mass_fuelflow_kg_s * self.lhv_fuel_j_kg) / 1000.0
        power_indicated_kw = fuel_energy_rate_kw * eta_indicated

        # 7. Friction & Pumping Mean Effective Pressure (FMEP in bar)
        fmep_bar = (self.fmep_base + self.fmep_rpm_coef * (rpm / 1000.0)) * mechanical_wear_factor
        # Pumping work (PMEP): difference between exhaust backpressure and intake MAP
        # Nominal exhaust backpressure ~ 1.15 * P_ambient or turbine inlet pressure
        pmep_bar = max(0.05, 0.15 * (rpm / 5000.0) ** 1.5)
        total_loss_mep_bar = fmep_bar + pmep_bar

        # Power loss from friction and pumping (kW)
        # Power = MEP (Pa) * V_d * (RPM/60) / 2
        power_loss_kw = (total_loss_mep_bar * 100000.0 * self.displacement_m3 * (rpm / 60.0) / 2.0) / 1000.0

        # 8. Brake Power (kW and HP)
        power_brake_kw = max(0.5, power_indicated_kw - power_loss_kw)
        power_brake_hp = power_brake_kw * 1.34102

        # 9. Brake Torque (N*m)
        omega = 2.0 * math.pi * (rpm / 60.0)
        torque_nm = (power_brake_kw * 1000.0) / omega

        # 10. Brake Mean Effective Pressure (BMEP in bar)
        bmep_bar = (power_brake_kw * 1000.0 * 2.0) / (self.displacement_m3 * (rpm / 60.0) * 100000.0)

        # 11. Brake Specific Fuel Consumption (BSFC in g/kWh)
        bsfc_g_kwh = (mass_fuelflow_kg_s * 3600.0 * 1000.0) / max(1.0, power_brake_kw)

        # 12. Brake Thermal Efficiency
        eta_brake_thermal = (power_brake_kw * 1000.0) / max(1.0, fuel_energy_rate_kw * 1000.0)

        return {
            "rpm": float(rpm),
            "map_pa": float(map_pa),
            "rho_intake": float(rho_intake),
            "eta_volumetric": float(eta_v),
            "mass_airflow_kg_s": float(mass_airflow_kg_s),
            "mass_fuelflow_kg_s": float(mass_fuelflow_kg_s),
            "fuel_flow_lph": float(fuelflow_lph),
            "power_indicated_kw": float(power_indicated_kw),
            "power_brake_kw": float(power_brake_kw),
            "power_brake_hp": float(power_brake_hp),
            "torque_nm": float(torque_nm),
            "bmep_bar": float(bmep_bar),
            "fmep_bar": float(fmep_bar),
            "bsfc_g_kwh": float(bsfc_g_kwh),
            "eta_brake_thermal": float(eta_brake_thermal),
            "fuel_energy_input_kw": float(fuel_energy_rate_kw)
        }

    def calculate_indicator_pv_cycle(
        self,
        map_pa: float,
        rpm: float = 5000.0,
        compression_ratio: float = 9.0,
        combustion_efficiency: float = 0.98,
        num_points: int = 72
    ) -> Dict[str, Any]:
        """
        Calculates crank-angle resolved (720 deg) in-cylinder pressure trace P(theta)
        and volume V(theta) to generate the thermodynamic P-V indicator diagram.
        """
        crank_radius_m = 0.061 / 2.0  # 30.5 mm
        conrod_length_m = 0.115       # 115 mm
        r_ratio = conrod_length_m / crank_radius_m
        v_disp_cyl = (self.displacement_m3 / 4.0)
        v_clearance = v_disp_cyl / (compression_ratio - 1.0)

        thetas_deg = np.linspace(-180.0, 180.0, num_points)
        thetas_rad = np.radians(thetas_deg)

        # 1. Slider-crank kinematic cylinder volume V(theta) in cm^3
        v_disp_profile = (v_disp_cyl / 2.0) * (
            r_ratio + 1.0 - np.cos(thetas_rad) - np.sqrt(np.maximum(0.01, r_ratio**2 - np.sin(thetas_rad)**2))
        )
        volume_cm3 = (v_clearance + v_disp_profile) * 1e6  # to cm^3

        # 2. Polytropic in-cylinder pressure P(theta)
        # Intake valve closure (IVC) at -140 deg; Spark advance (SA) at -22 deg; TDC at 0 deg
        p_intake_bar = map_pa / 100000.0
        p_exhaust_bar = 1.08  # slight backpressure past turbine

        # Polytropic exponents (compression k_c ~ 1.33, expansion k_e ~ 1.28)
        k_c = 1.33
        k_e = 1.28

        # Compression peak at TDC before combustion
        v_max = (v_clearance + v_disp_cyl) * 1e6
        p_comp_tdc = p_intake_bar * ((v_max / (v_clearance * 1e6)) ** k_c)

        # Firing pressure rise via Wiebe function
        delta_p_comb = p_comp_tdc * 1.85 * combustion_efficiency

        pressures_bar = []
        for th, vol in zip(thetas_deg, volume_cm3):
            if th <= -22.0:
                # Compression stroke
                p = p_intake_bar * ((v_max / vol) ** k_c)
            elif -22.0 < th <= 45.0:
                # Active combustion phase (Wiebe mass fraction burned)
                burn_fraction = 0.5 * (1.0 - math.cos(math.pi * (th - (-22.0)) / (45.0 - (-22.0))))
                p_motored = p_intake_bar * ((v_max / vol) ** k_c)
                p = p_motored + delta_p_comb * (burn_fraction ** 1.3)
            else:
                # Expansion stroke
                v_exp_ref = (v_clearance + (v_disp_cyl / 2.0) * (
                    r_ratio + 1.0 - math.cos(math.radians(45.0)) - math.sqrt(max(0.01, r_ratio**2 - math.sin(math.radians(45.0))**2))
                )) * 1e6
                p_peak_ref = p_intake_bar * ((v_max / v_exp_ref) ** k_c) + delta_p_comb
                p = max(p_exhaust_bar, p_peak_ref * ((v_exp_ref / vol) ** k_e))

            pressures_bar.append(round(float(p), 2))

        peak_pressure_bar = max(pressures_bar)

        return {
            "volume_cm3": [round(float(v), 1) for v in volume_cm3],
            "pressure_bar": pressures_bar,
            "peak_cylinder_pressure_bar": round(peak_pressure_bar, 1),
            "crank_angles_deg": [round(float(th), 1) for th in thetas_deg],
            "indicated_work_joules_per_cyl": round(float(peak_pressure_bar * 1e5 * v_disp_cyl * 0.38), 1)
        }

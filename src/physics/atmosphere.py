"""
International Standard Atmosphere (ISA) and Non-Standard Atmosphere Model
Calculates pressure, temperature, density, and sound speed as a function of altitude
and ambient temperature deviations (delta ISA) for MALE UAV flight envelopes up to 35,000 ft.
"""

import math
from typing import Dict, Any

# Physical constants
G0 = 9.80665              # Gravitational acceleration, m/s^2
R_AIR = 287.05287         # Gas constant for dry air, J/(kg·K)
GAMMA = 1.4               # Specific heat ratio for air
LAPSE_RATE = -0.0065      # Temperature lapse rate in troposphere, K/m
T0_ISA = 288.15           # Sea-level standard temperature, K (15 °C)
P0_ISA = 101325.0         # Sea-level standard pressure, Pa (29.921 inHg)
RHO0_ISA = 1.225          # Sea-level standard air density, kg/m^3
TROPOPAUSE_ALT_M = 11000.0# Altitude of tropopause, m (approx 36,089 ft)


def feet_to_meters(feet: float) -> float:
    return feet * 0.3048


def meters_to_feet(meters: float) -> float:
    return meters / 0.3048


def pa_to_inhg(pa: float) -> float:
    return pa / 3386.389


def inhg_to_pa(inhg: float) -> float:
    return inhg * 3386.389


class AtmosphereModel:
    """
    ISA tropospheric atmospheric profile with customizable delta-ISA for extreme weather conditions
    (e.g., desert operations +45°C or high-altitude cold soak -35°C).
    """

    def __init__(self, delta_isa_c: float = 0.0):
        self.delta_isa_c = float(delta_isa_c)

    def calculate(self, altitude_ft: float, ambient_temp_c: float = None) -> Dict[str, float]:
        """
        Compute atmospheric state at given geometric altitude.
        If ambient_temp_c is provided, delta_isa is dynamically derived from it.
        """
        alt_m = max(0.0, feet_to_meters(altitude_ft))
        
        # ISA standard temperature at this altitude
        if alt_m <= TROPOPAUSE_ALT_M:
            t_isa = T0_ISA + LAPSE_RATE * alt_m
            p_ratio = (t_isa / T0_ISA) ** (-G0 / (R_AIR * LAPSE_RATE))
        else:
            t_isa_tropo = T0_ISA + LAPSE_RATE * TROPOPAUSE_ALT_M
            t_isa = t_isa_tropo
            p_ratio_tropo = (t_isa_tropo / T0_ISA) ** (-G0 / (R_AIR * LAPSE_RATE))
            p_ratio = p_ratio_tropo * math.exp(-G0 * (alt_m - TROPOPAUSE_ALT_M) / (R_AIR * t_isa))

        p_pa = P0_ISA * p_ratio

        if ambient_temp_c is not None:
            t_actual_k = ambient_temp_c + 273.15
            effective_delta_isa = t_actual_k - t_isa
        else:
            effective_delta_isa = self.delta_isa_c
            t_actual_k = t_isa + effective_delta_isa

        density_kg_m3 = p_pa / (R_AIR * t_actual_k)
        speed_of_sound_ms = math.sqrt(GAMMA * R_AIR * t_actual_k)
        density_ratio = density_kg_m3 / RHO0_ISA

        return {
            "altitude_ft": float(altitude_ft),
            "altitude_m": float(alt_m),
            "pressure_pa": float(p_pa),
            "pressure_inhg": float(pa_to_inhg(p_pa)),
            "temperature_k": float(t_actual_k),
            "temperature_c": float(t_actual_k - 273.15),
            "density_kg_m3": float(density_kg_m3),
            "density_ratio": float(density_ratio),
            "speed_of_sound_ms": float(speed_of_sound_ms),
            "delta_isa_c": float(effective_delta_isa)
        }

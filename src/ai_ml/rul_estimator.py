"""
Remaining Useful Life (RUL) Prognostics and Stochastic Degradation Tracker
Estimates remaining flight operating hours to maintenance thresholds using physics-informed
Wiener process degradation modeling with probabilistic bounds (P10, P50, P90).
"""

import math
from typing import Dict, Any, Tuple
from ..core.state import CompositeEngineHealth, RULPrognosis

class RULPrognosticsEstimator:
    """
    Stochastic prognostic model for aero piston engine life consumption.
    Tracks progressive degradation across thermal, mechanical, and lubrication sub-systems.
    """
    def __init__(self, initial_operating_hours: float = 340.0, nominal_tbo_hours: float = 1800.0):
        self.operating_hours = float(initial_operating_hours)
        self.nominal_tbo_hours = float(nominal_tbo_hours)
        
        # Degradation thresholds
        self.health_caution_threshold = 0.65
        self.health_critical_threshold = 0.35

        # Baseline nominal degradation rate per operating hour (1.0 -> 0.35 over 1800 hrs = ~0.000361 / hr)
        self.base_drift_rate = (1.0 - self.health_critical_threshold) / self.nominal_tbo_hours
        self.diffusion_sigma = 0.00015  # Brownian stochastic diffusion

        self.filtered_drift_rate = self.base_drift_rate

    def update(self, health: CompositeEngineHealth, dt_flight_hours: float = 0.000277) -> RULPrognosis:
        """
        Update accumulated engine flight hours and evaluate probabilistic RUL.
        dt_flight_hours default is 1 second = 1/3600 hr = ~0.000277 hr.
        """
        self.operating_hours += dt_flight_hours
        ehi = max(0.01, min(1.0, health.overall_health_index))

        # Dynamic stress factor based on current health deficits
        # If health has degraded, wear rate accelerates exponentially (Arrhenius / Paris law style)
        health_deficit = max(0.0, 1.0 - ehi)
        stress_multiplier = 1.0 + 4.5 * (health_deficit ** 1.8)
        
        # Subsystem-specific stress penalties
        sub = health.subsystem_scores
        if sub.thermal_cht < 70.0:
            stress_multiplier += 1.5 * ((70.0 - sub.thermal_cht) / 70.0)
        if sub.lubrication_system < 70.0:
            stress_multiplier += 2.8 * ((70.0 - sub.lubrication_system) / 70.0)
        if sub.combustion_stability < 60.0:
            stress_multiplier += 2.0 * ((60.0 - sub.combustion_stability) / 60.0)

        instant_drift = self.base_drift_rate * stress_multiplier

        # Smooth drift rate filter
        self.filtered_drift_rate = 0.98 * self.filtered_drift_rate + 0.02 * instant_drift

        # Remaining health budget to thresholds
        budget_to_caution = max(0.0, ehi - self.health_caution_threshold)
        budget_to_critical = max(0.0, ehi - self.health_critical_threshold)

        # First-hitting-time (FHT) median estimate (P50 in hours)
        rul_critical_median = budget_to_critical / max(1e-6, self.filtered_drift_rate)
        rul_caution_median = budget_to_caution / max(1e-6, self.filtered_drift_rate)

        # Probabilistic confidence bounds (Inverse Gaussian distribution quantile approximation)
        uncertainty_factor = (self.diffusion_sigma / max(1e-6, self.filtered_drift_rate)) * math.sqrt(max(1.0, rul_critical_median))
        margin_90 = 1.282 * uncertainty_factor * 12.0  # 90% confidence spread
        
        rul_p10 = max(0.0, rul_critical_median - margin_90)
        rul_p90 = rul_critical_median + margin_90

        # Identify primary limiting subsystem
        subsystem_dict = {
            "Thermal CHT / Valve Guides": sub.thermal_cht,
            "Exhaust Gas Path / Valves": sub.exhaust_gas_path,
            "Fuel Injection / Delivery": sub.fuel_injection,
            "Crankshaft Journal Lubrication": sub.lubrication_system,
            "Turbocharger Boost Control": sub.turbo_charge,
            "Coolant Loop Heat Transfer": sub.cooling_system,
            "Combustion Stability / Ignition": sub.combustion_stability,
            "Electrical Alternator Subsystem": sub.electrical_bus
        }
        limiting_subsystem = min(subsystem_dict, key=subsystem_dict.get)

        # Trend categorization
        if ehi >= 0.88 and stress_multiplier < 1.3:
            trend_category = "STABLE_NOMINAL"
        elif ehi >= 0.70 and stress_multiplier < 2.0:
            trend_category = "LINEAR_NORMAL"
        elif ehi >= 0.45:
            trend_category = "ACCELERATING_WEAR"
        else:
            trend_category = "IMMINENT_CRITICAL"

        # Mission completion probability for an illustrative 14-hour MALE UAV endurance sortie
        mission_hours_remaining = 14.0
        if rul_critical_median <= 0.1:
            mission_prob = 0.0
        else:
            # Gaussian / log-normal tail probability
            z_score = (rul_critical_median - mission_hours_remaining) / max(1.0, margin_90)
            mission_prob = min(99.9, max(5.0, 50.0 + 49.9 * math.erf(z_score / math.sqrt(2.0))))

        return RULPrognosis(
            current_operating_hours=round(self.operating_hours, 1),
            rul_hours_median=round(rul_critical_median, 1),
            rul_hours_p10=round(rul_p10, 1),
            rul_hours_p90=round(rul_p90, 1),
            primary_limiting_subsystem=limiting_subsystem,
            degradation_trend=trend_category,
            time_to_caution_hours=round(rul_caution_median, 1),
            time_to_critical_hours=round(rul_critical_median, 1),
            mission_completion_probability=round(mission_prob, 1)
        )

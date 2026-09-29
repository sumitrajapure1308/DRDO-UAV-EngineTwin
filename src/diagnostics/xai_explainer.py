"""
Explainable AI (XAI) Attribution Module for Engine Diagnostics
Decomposes multivariate anomaly scores and fault detections into quantified feature contributions,
providing transparent physical rationale for UAV operators and propulsion engineers.
"""

from typing import Dict, Any, List
import math
from ..core.state import TelemetryFrame, DigitalTwinEstimate

class XAIExplainer:
    """
    Computes normalized feature importance attributions for diagnostic events.
    Uses normalized Mahalanobis/z-score deviations of physics residuals.
    """

    def __init__(self):
        # Baseline expected variance for normalized scaling
        self.sigma_baseline = {
            "residual_cht_1": 4.5,
            "residual_cht_2": 4.5,
            "residual_cht_3": 4.5,
            "residual_cht_4": 4.5,
            "residual_egt_1": 25.0,
            "residual_egt_2": 25.0,
            "residual_egt_3": 25.0,
            "residual_egt_4": 25.0,
            "residual_oil_pressure": 0.45,
            "residual_oil_temp": 4.0,
            "residual_map": 2500.0,
            "residual_fuel_flow": 1.2,
            "residual_vibration": 0.35,
            "vibration_0_5x": 0.15,
            "vibration_1_0x": 0.35,
            "knock_intensity": 0.15,
            "bus_voltage": 0.6
        }

    def explain(self, telem: TelemetryFrame, twin: DigitalTwinEstimate) -> Dict[str, Any]:
        """
        Compute percentage contribution of each physics residual and sensor channel
        towards the current system deviation.
        """
        deviations = {
            "Cylinder 1 CHT Residual": (abs(twin.residual_cht_1_c) / self.sigma_baseline["residual_cht_1"]) ** 2,
            "Cylinder 2 CHT Residual": (abs(twin.residual_cht_2_c) / self.sigma_baseline["residual_cht_2"]) ** 2,
            "Cylinder 3 CHT Residual": (abs(twin.residual_cht_3_c) / self.sigma_baseline["residual_cht_3"]) ** 2,
            "Cylinder 4 CHT Residual": (abs(twin.residual_cht_4_c) / self.sigma_baseline["residual_cht_4"]) ** 2,
            "Cylinder 1 EGT Residual": (abs(twin.residual_egt_1_c) / self.sigma_baseline["residual_egt_1"]) ** 2,
            "Cylinder 2 EGT Residual": (abs(twin.residual_egt_2_c) / self.sigma_baseline["residual_egt_2"]) ** 2,
            "Cylinder 3 EGT Residual": (abs(twin.residual_egt_3_c) / self.sigma_baseline["residual_egt_3"]) ** 2,
            "Cylinder 4 EGT Residual": (abs(twin.residual_egt_4_c) / self.sigma_baseline["residual_egt_4"]) ** 2,
            "Oil Pressure Residual": (abs(twin.residual_oil_pressure_bar) / self.sigma_baseline["residual_oil_pressure"]) ** 2,
            "Oil Temperature Residual": (abs(twin.residual_oil_temp_c) / self.sigma_baseline["residual_oil_temp"]) ** 2,
            "MAP Boost Residual": (abs(twin.residual_map_pa) / self.sigma_baseline["residual_map"]) ** 2,
            "Fuel Flow Residual": (abs(twin.residual_fuel_flow_lph) / self.sigma_baseline["residual_fuel_flow"]) ** 2,
            "Vibration RMS Residual": (abs(twin.residual_vibration_rms_g) / self.sigma_baseline["residual_vibration"]) ** 2,
            "Vibration 0.5x Order (Misfire)": ((telem.crest_factor - 2.4) / 0.5) ** 2 if telem.crest_factor > 2.4 else 0.0,
            "Electrical Bus Stability": (abs(telem.bus_voltage_v - 28.0) / self.sigma_baseline["bus_voltage"]) ** 2
        }

        total_sum = sum(deviations.values())
        if total_sum < 1e-6:
            # All nominal
            attributions = {k: 0.0 for k in deviations}
            top_drivers = []
        else:
            attributions = {k: round((v / total_sum) * 100.0, 1) for k, v in deviations.items()}
            # Sort top 5 explanatory drivers
            sorted_drivers = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
            top_drivers = [{"feature": k, "contribution_pct": v} for k, v in sorted_drivers if v > 2.0][:5]

        return {
            "total_deviation_score": round(total_sum, 2),
            "feature_attributions_pct": attributions,
            "top_drivers": top_drivers
        }

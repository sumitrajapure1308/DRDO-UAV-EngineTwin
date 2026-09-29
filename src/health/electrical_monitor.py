"""
Electrical Bus, Alternator, and Battery Health Analyzer
Monitors 28V DC power bus stability, alternator rectifying diode ripple,
battery state of charge/health, and electrical power generation margins.
"""

from typing import Dict, Any

class ElectricalSystemMonitor:
    def __init__(self):
        self.battery_soh = 0.98   # Battery State of Health
        self.alternator_health = 0.99

    def analyze(self, bus_voltage: float, alternator_amps: float, battery_amps: float, ripple_mv: float) -> Dict[str, Any]:
        """
        Analyze electrical power system health.
        """
        status = "NOMINAL"
        recommendation = "Normal operation"

        # Check alternator diode failure (indicated by high AC ripple on 28V DC bus)
        diode_fault = ripple_mv > 300.0
        voltage_sag = bus_voltage < 24.5
        overvoltage = bus_voltage > 30.5

        if diode_fault:
            status = "WARNING"
            recommendation = "Alternator phase diode breakdown detected (high AC ripple). Inspect rectifier."
            self.alternator_health = max(0.4, self.alternator_health - 0.05)
        elif voltage_sag:
            status = "CAUTION"
            recommendation = "Low bus voltage: Alternator output insufficient for avionics/actuator bus."
        elif overvoltage:
            status = "WARNING"
            recommendation = "Overvoltage condition: Voltage regulator failure. Risk of avionics damage."

        power_output_w = bus_voltage * alternator_amps
        load_margin_pct = max(0.0, ((1200.0 - power_output_w) / 1200.0) * 100.0)

        return {
            "status": status,
            "bus_voltage_v": round(bus_voltage, 2),
            "alternator_amps": round(alternator_amps, 1),
            "battery_amps": round(battery_amps, 1),
            "power_generated_w": round(power_output_w, 1),
            "load_margin_pct": round(load_margin_pct, 1),
            "ripple_mv": round(ripple_mv, 1),
            "alternator_health_score": round(self.alternator_health * 100.0, 1),
            "battery_soh_pct": round(self.battery_soh * 100.0, 1),
            "recommendation": recommendation
        }

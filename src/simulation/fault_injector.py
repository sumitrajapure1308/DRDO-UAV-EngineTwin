"""
Real-Time Dynamic Fault Injector for Aero Engine Simulation
Provides programmatic and interactive fault injection for testing and validating
AI diagnostics, predictive alerts, and RUL prognostics during live flight.
"""

from typing import Dict, Any, List, Optional

class FaultInjector:
    def __init__(self):
        self.active_faults: Dict[str, Dict[str, Any]] = {}

    def inject_fault(self, fault_type: str, severity: float = 0.5, target_cylinder: int = 1, params: Dict[str, Any] = None):
        """
        Inject a specific aero engine fault.
        fault_type in:
          - "misfire"
          - "injector_abnormal"
          - "cooling_degradation"
          - "lubrication_issue"
          - "sensor_drift"
          - "combustion_instability"
          - "overheating_trend"
          - "abnormal_vibration"
        """
        severity = max(0.0, min(1.0, float(severity)))
        target_cylinder = max(1, min(4, int(target_cylinder)))
        self.active_faults[fault_type] = {
            "severity": severity,
            "target_cylinder": target_cylinder,
            "params": params or {}
        }

    def clear_fault(self, fault_type: str):
        if fault_type in self.active_faults:
            del self.active_faults[fault_type]

    def clear_all(self):
        self.active_faults.clear()

    def get_state(self) -> Dict[str, Any]:
        return dict(self.active_faults)

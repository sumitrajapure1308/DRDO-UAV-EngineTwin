"""
Autonomous Prescriptive Maintenance Advisory Engine
Translates real-time diagnostic alerts, subsystem health scores, and RUL prognostics
into standardized aviation maintenance advisories and prioritized work orders.
"""

from typing import List, Dict, Any
from ..core.state import FaultAlert, CompositeEngineHealth, RULPrognosis

class MaintenanceAdvisoryEngine:
    def __init__(self):
        self.advisory_history: List[Dict[str, Any]] = []

    def generate_advisories(
        self,
        alerts: List[FaultAlert],
        health: CompositeEngineHealth,
        prognosis: RULPrognosis
    ) -> List[Dict[str, Any]]:
        """
        Synthesize prioritized maintenance recommendations.
        """
        advisories: List[Dict[str, Any]] = []

        # 1. Critical & Warning Active Alerts
        for alert in alerts:
            priority = "URGENT_AOG" if alert.severity == "CRITICAL" else "PRE_FLIGHT_ACTION"
            adv = {
                "advisory_id": f"ADV-{alert.alert_id}",
                "priority": priority,
                "urgency_level": alert.severity,
                "ata_system": alert.fault_code.split("-")[0] if "-" in alert.fault_code else "ATA 71",
                "component": alert.title,
                "findings": alert.description,
                "prescribed_action": alert.recommended_action,
                "affected_cylinder": alert.affected_cylinder,
                "estimated_downtime_hours": 3.5 if alert.severity == "CRITICAL" else 1.5
            }
            advisories.append(adv)

        # 2. Prognostic Wear & Subsystem Degradation Advisories
        sub = health.subsystem_scores
        
        # Check lubrication degradation
        if sub.lubrication_system < 80.0 and not any(a.subsystem == "lubrication_system" for a in alerts):
            advisories.append({
                "advisory_id": "ADV-PROG-LUBE-01",
                "priority": "SCHEDULED_INSPECTION",
                "urgency_level": "CAUTION",
                "ata_system": "ATA 79",
                "component": "Engine Lubrication Circuit & Sump",
                "findings": f"Lubrication Health Index degraded to {sub.lubrication_system}%. Operating hours: {prognosis.current_operating_hours} hrs.",
                "prescribed_action": "Draw 100ml oil sample for Spectrometric Oil Analysis (SOAP). Replace oil filter element and check magnetic chip detector for ferrous particles.",
                "affected_cylinder": None,
                "estimated_downtime_hours": 1.0
            })

        # Check thermal CHT degradation
        if sub.thermal_cht < 80.0 and not any(a.subsystem in ["thermal_cht", "cooling_system"] for a in alerts):
            advisories.append({
                "advisory_id": "ADV-PROG-THERM-01",
                "priority": "SCHEDULED_INSPECTION",
                "urgency_level": "CAUTION",
                "ata_system": "ATA 75",
                "component": "Cylinder Head Liquid Cooling Jackets",
                "findings": f"Thermal health score at {sub.thermal_cht}%. Elevated cylinder-to-cylinder thermal resistance observed.",
                "prescribed_action": "Flush coolant loop with 50/50 water-glycol mixture, clean radiator external matrix fins of bug/dust debris, pressure test cooling cap to 1.2 bar.",
                "affected_cylinder": None,
                "estimated_downtime_hours": 2.0
            })

        # Check RUL horizon
        if prognosis.rul_hours_median < 150.0:
            advisories.append({
                "advisory_id": "ADV-PROG-TBO-01",
                "priority": "SCHEDULED_OVERHAUL",
                "urgency_level": "WARNING" if prognosis.rul_hours_median < 50.0 else "ADVISORY",
                "ata_system": "ATA 72",
                "component": "Complete Powerplant Overhaul",
                "findings": f"Remaining Useful Life (P50) is {prognosis.rul_hours_median} flight hours. Limiting subsystem: {prognosis.primary_limiting_subsystem}.",
                "prescribed_action": "Schedule depot-level overhaul and reserve replacement aero engine module. Perform differential cylinder compression test.",
                "affected_cylinder": None,
                "estimated_downtime_hours": 24.0
            })

        # If everything is completely nominal
        if not advisories:
            advisories.append({
                "advisory_id": "ADV-NOMINAL-00",
                "priority": "ROUTINE_MONITORING",
                "urgency_level": "NOMINAL",
                "ata_system": "ATA 71",
                "component": "All Engine Subsystems",
                "findings": f"All 8 monitored subsystems nominal. Composite Health Index: {round(health.overall_health_index * 100, 1)}%.",
                "prescribed_action": "Continue scheduled flight operations. Next 50-hour routine check due at " + str(round((int(prognosis.current_operating_hours / 50) + 1) * 50, 0)) + " flight hours.",
                "affected_cylinder": None,
                "estimated_downtime_hours": 0.0
            })

        self.advisory_history = advisories
        return advisories

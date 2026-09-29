"""
Aero Engine Fault Catalog and Standards Taxonomy
Maps diagnostic codes to aerospace ATA chapters (ATA 71, 72, 73, 75, 77, 79),
MIL-STD severity levels, and standardized maintenance actions.
"""

from typing import Dict, Any

FAULT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "MISFIRE_DETECTED": {
        "code": "ATA72-MISF-01",
        "ata_chapter": "ATA 72 (Engine Reciprocating)",
        "subsystem": "combustion_stability",
        "default_severity": "CRITICAL",
        "title": "Cylinder Combustion Misfire",
        "description": "Loss of combustion event detected via abrupt localized EGT quenching and subharmonic 0.5x vibration spike.",
        "recommended_action": "Execute immediate power check; inspect spark plugs, secondary ignition coil, and fuel injector on affected cylinder."
    },
    "INJECTOR_ABNORMAL": {
        "code": "ATA73-INJ-02",
        "ata_chapter": "ATA 73 (Engine Fuel and Control)",
        "subsystem": "fuel_injection",
        "default_severity": "WARNING",
        "title": "Fuel Injector Delivery Abnormality",
        "description": "Cylinder-to-cylinder fuel delivery asymmetry exceeding operational tolerance (lean/rich deviation).",
        "recommended_action": "Borescope check; perform bench ultrasonic cleaning and spray pattern flow calibration on affected injector."
    },
    "COOLING_DEGRADATION": {
        "code": "ATA75-COOL-01",
        "ata_chapter": "ATA 75 (Engine Air / Cooling)",
        "subsystem": "cooling_system",
        "default_severity": "WARNING",
        "title": "Cooling System Thermal Heat Rejection Degradation",
        "description": "CHT rise rate and heat transfer residual indicate coolant flow restriction or radiator matrix fouling.",
        "recommended_action": "Check coolant system expansion tank level, purge trapped air, inspect radiator ram air duct and coolant pump impeller."
    },
    "LUBRICATION_ISSUE": {
        "code": "ATA79-LUBE-01",
        "ata_chapter": "ATA 79 (Engine Oil)",
        "subsystem": "lubrication_system",
        "default_severity": "CRITICAL",
        "title": "Lubrication Oil Pressure Degradation",
        "description": "Oil pressure below hydrodynamic bearing threshold relative to operating temperature. High friction risk.",
        "recommended_action": "AOG Alert: Immediate engine power reduction if airborne. Inspect oil filter for metal particulate, check pressure relief valve and oil lines."
    },
    "SENSOR_DRIFT_FAILURE": {
        "code": "ATA77-SENS-01",
        "ata_chapter": "ATA 77 (Engine Indicating)",
        "subsystem": "sensors",
        "default_severity": "CAUTION",
        "title": "Sensor Calibration Drift / Plausibility Discrepancy",
        "description": "Sensor output deviates from digital twin analytical physics redundancy while coupled thermodynamic variables remain nominal.",
        "recommended_action": "Perform sensor electrical loop calibration and wiring harness continuity check; replace defective probe."
    },
    "COMBUSTION_INSTABILITY": {
        "code": "ATA72-COMB-01",
        "ata_chapter": "ATA 72 (Engine Reciprocating)",
        "subsystem": "combustion_stability",
        "default_severity": "WARNING",
        "title": "Combustion Knock / Cyclic Instability",
        "description": "Cyclic combustion variability and high-frequency acoustic/vibration detonation signatures detected.",
        "recommended_action": "Retard ignition timing by 2-3 degrees; verify fuel octane rating and check intake air temperature intercooler efficiency."
    },
    "OVERHEATING_TREND": {
        "code": "ATA75-OHT-02",
        "ata_chapter": "ATA 75 (Engine Air / Cooling)",
        "subsystem": "thermal_cht",
        "default_severity": "CAUTION",
        "title": "Predictive Overheating Trajectory",
        "description": "Thermal derivative extrapolation projects CHT/Oil temp will breach redline limits within 180 seconds under current power setting.",
        "recommended_action": "Lower climb rate or increase airspeed by 10 kts to boost ram air mass flow; enrich mixture to cool combustion chambers."
    },
    "ABNORMAL_VIBRATION": {
        "code": "ATA72-VIB-02",
        "ata_chapter": "ATA 72 (Engine Reciprocating)",
        "subsystem": "combustion_stability",
        "default_severity": "WARNING",
        "title": "Abnormal Mechanical Vibration Signature",
        "description": "Harmonic order tracking reveals unbalance at 1.0x shaft frequency or bearing degradation in high-frequency spectrum.",
        "recommended_action": "Perform dynamic propeller track and balance; inspect engine shock mounts and perform oil spectrographic wear debris analysis."
    }
}

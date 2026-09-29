"""
Core Data Contracts and Engine State Definitions
Pydantic data models for real-time telemetry, digital twin physics estimates,
subsystem health indices, prognostic RUL estimates, and fault alerts.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import time

class TelemetryFrame(BaseModel):
    timestamp_s: float = Field(default_factory=time.time)
    mission_time_s: float = 0.0
    mission_phase: str = "CRUISE"
    
    # Atmospheric / Flight parameters
    altitude_ft: float = 5000.0
    ambient_temp_c: float = 15.0
    ambient_pressure_pa: float = 84307.0
    airspeed_kts: float = 90.0

    # Primary propulsion dynamics
    throttle_pct: float = 70.0
    rpm: float = 5000.0
    map_inhg: float = 30.5
    map_pa: float = 103285.0
    
    # Thermal parameters
    cht_1_c: float = 95.0
    cht_2_c: float = 96.0
    cht_3_c: float = 98.0
    cht_4_c: float = 97.0
    cht_avg_c: float = 96.5
    cht_spread_c: float = 3.0

    egt_1_c: float = 760.0
    egt_2_c: float = 765.0
    egt_3_c: float = 770.0
    egt_4_c: float = 762.0
    egt_avg_c: float = 764.2
    egt_spread_c: float = 10.0

    coolant_temp_c: float = 85.0

    # Lubrication
    oil_pressure_bar: float = 4.2
    oil_temp_c: float = 90.0

    # Fuel system & mixture
    fuel_flow_lph: float = 24.5
    fuel_pressure_bar: float = 3.0
    air_fuel_ratio: float = 14.2
    injection_timing_deg_btdc: float = 24.0
    injection_pulse_width_ms: float = 6.2
    ignition_timing_deg_btdc: float = 26.0

    # Turbocharger
    wastegate_duty_pct: float = 45.0
    compressor_ratio: float = 1.25
    intercooler_exit_temp_c: float = 38.0

    # Mechanical / Vibration
    vibration_rms_g: float = 1.45
    vibration_peak_g: float = 3.2
    vibration_x_rms_g: float = 1.5
    vibration_y_rms_g: float = 1.3
    vibration_z_rms_g: float = 0.95
    crest_factor: float = 2.4
    spectral_bins: List[Dict[str, Any]] = Field(default_factory=list)

    # Electrical System
    bus_voltage_v: float = 28.1
    alternator_current_a: float = 24.5
    battery_current_a: float = 1.2
    voltage_ripple_mv: float = 85.0


class DigitalTwinEstimate(BaseModel):
    timestamp_s: float
    
    # Physics predicted states
    twin_map_pa: float
    twin_map_inhg: float
    twin_power_brake_kw: float
    twin_power_brake_hp: float
    twin_bmep_bar: float
    twin_bsfc_g_kwh: float
    twin_eta_thermal: float
    twin_fuel_flow_lph: float
    
    twin_cht_1_c: float
    twin_cht_2_c: float
    twin_cht_3_c: float
    twin_cht_4_c: float
    twin_egt_1_c: float
    twin_egt_2_c: float
    twin_egt_3_c: float
    twin_egt_4_c: float
    
    twin_coolant_temp_c: float
    twin_oil_temp_c: float
    twin_oil_pressure_bar: float
    twin_vibration_rms_g: float

    # Real vs Twin Physics Residuals (y_meas - y_pred)
    residual_map_pa: float
    residual_cht_1_c: float
    residual_cht_2_c: float
    residual_cht_3_c: float
    residual_cht_4_c: float
    residual_egt_1_c: float
    residual_egt_2_c: float
    residual_egt_3_c: float
    residual_egt_4_c: float
    residual_fuel_flow_lph: float
    residual_oil_pressure_bar: float
    residual_oil_temp_c: float
    residual_vibration_rms_g: float
    
    composite_residual_norm: float


class SubsystemHealth(BaseModel):
    thermal_cht: float = 100.0          # 0 - 100%
    exhaust_gas_path: float = 100.0     # 0 - 100%
    fuel_injection: float = 100.0       # 0 - 100%
    lubrication_system: float = 100.0   # 0 - 100%
    turbo_charge: float = 100.0         # 0 - 100%
    cooling_system: float = 100.0       # 0 - 100%
    combustion_stability: float = 100.0 # 0 - 100%
    electrical_bus: float = 100.0       # 0 - 100%


class CompositeEngineHealth(BaseModel):
    overall_health_index: float = 1.0   # 1.0 = brand new, 0.0 = failure
    subsystem_scores: SubsystemHealth = Field(default_factory=SubsystemHealth)
    degradation_rate_per_hr: float = 0.0001
    mission_reliability_pct: float = 99.8
    thermal_margin_c: float = 38.0
    oil_pressure_margin_bar: float = 2.2


class FaultAlert(BaseModel):
    alert_id: str
    timestamp_s: float
    fault_code: str
    subsystem: str
    severity: str                       # NOMINAL, ADVISORY, CAUTION, WARNING, CRITICAL
    title: str
    description: str
    affected_cylinder: Optional[int] = None
    root_cause_evidence: Dict[str, Any] = Field(default_factory=dict)
    recommended_action: str = ""
    is_active: bool = True


class RULPrognosis(BaseModel):
    current_operating_hours: float
    rul_hours_median: float             # P50 estimate
    rul_hours_p10: float                # Pessimistic lower bound (90% confidence)
    rul_hours_p90: float                # Optimistic upper bound
    primary_limiting_subsystem: str
    degradation_trend: str              # "STABLE_NOMINAL", "LINEAR_NORMAL", "ACCELERATING_WEAR", "IMMINENT_CRITICAL"
    time_to_caution_hours: float
    time_to_critical_hours: float
    mission_completion_probability: float # for remaining mission duration


class SensorFusionState(BaseModel):
    estimated_p_max_bar: float = 65.0
    p_max_confidence_pm_bar: float = 4.2
    estimated_t_core_k: float = 2100.0
    estimated_trapped_mass_mg: float = 420.0
    estimated_friction_torque_nm: float = 14.5
    degradation_index: float = 1.0
    sensor_fusion_mahalanobis_distance: float = 0.85
    sensor_consistency_status: str = "CONSISTENT"


class MissionReliabilityEnvelope(BaseModel):
    mission_directive: str = "GO_FULL_MISSION"
    max_safe_throttle_pct: float = 100.0
    derate_reasons: List[str] = Field(default_factory=list)
    total_reachable_range_nm: float = 380.0
    powered_range_nm: float = 365.0
    deadstick_glide_range_nm: float = 15.0
    endurance_remaining_hours: float = 3.45
    operational_advisory: str = "Nominal mission execution"
    recovery_bases: List[Dict[str, Any]] = Field(default_factory=list)


class EngineFullStatePacket(BaseModel):
    telemetry: TelemetryFrame
    digital_twin: DigitalTwinEstimate
    health: CompositeEngineHealth
    active_alerts: List[FaultAlert] = Field(default_factory=list)
    prognosis: RULPrognosis
    xai_attributions: Dict[str, float] = Field(default_factory=dict)
    sensor_fusion: Optional[Any] = None
    reliability_envelope: Optional[Any] = None
    sensor_redundancy: Optional[Any] = None
    fadec_state: Optional[Any] = None


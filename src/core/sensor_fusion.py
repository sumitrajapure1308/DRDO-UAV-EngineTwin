"""
DRDO MALE UAV Aero Piston Engine Digital Twin
Extended Kalman Filter (EKF) Sensor Fusion & Unmeasured State Estimator

Implements a non-linear Extended Kalman Filter fusing asynchronous telemetry measurements
(RPM, MAP, Fuel Flow, CHT, EGT, Oil P, Vibration) with internal thermodynamic state equations
to estimate unmeasured in-cylinder parameters:
- In-cylinder peak firing pressure (P_max)
- Core combustion flame temperature (T_core)
- Trapped cylinder charge mass (m_trapped)
- Internal mechanical friction torque (tau_friction)
- Dynamic component degradation multiplier
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
from .state import TelemetryFrame

class EngineSensorFusionEKF:
    """
    Extended Kalman Filter for aero propulsion sensor fusion and state estimation.
    State vector x = [P_max_bar, T_core_k, m_trapped_mg, tau_friction_nm, degradation_factor]^T
    """

    def __init__(self):
        # 5 State variables:
        # x[0]: Peak cylinder pressure (bar)
        # x[1]: In-cylinder core combustion temperature (K)
        # x[2]: Trapped cylinder air mass per cycle (mg/cyl)
        # x[3]: Mechanical friction torque (N*m)
        # x[4]: Engine degradation index (1.0 = nominal, > 1.0 = degraded)
        self.state_dim = 5
        self.x = np.array([65.0, 2100.0, 420.0, 14.5, 1.0], dtype=float)

        # State error covariance matrix P
        self.P = np.diag([25.0, 10000.0, 400.0, 4.0, 0.05])

        # Process noise covariance Q (system dynamics uncertainty)
        self.Q = np.diag([0.8, 45.0, 1.5, 0.08, 0.0001])

        # Measurement noise covariance R (sensor noise variance)
        # Measurements: [MAP_bar, FuelFlow_lph, CHT_avg_c, EGT_avg_c, OilP_bar, Vib_RMS_g]
        self.meas_dim = 6
        self.R = np.diag([
            0.05**2,    # MAP sensor (+-0.05 bar)
            1.2**2,     # Fuel flow turbine sensor (+-1.2 L/h)
            3.5**2,     # CHT thermocouple (+-3.5 C)
            12.0**2,    # EGT thermocouple (+-12.0 C)
            0.15**2,    # Oil pressure transducer (+-0.15 bar)
            0.10**2     # Accelerometer RMS (+-0.10 g)
        ])

        self.last_timestamp_s = None
        self.innovation_history = []
        self.mahalanobis_distance = 0.0

    def predict(self, rpm: float, throttle_pct: float, dt_s: float = 0.1):
        """
        Time-update (prediction step) using non-linear engine physical continuity.
        """
        dt = max(0.01, min(0.5, float(dt_s)))
        rpm = max(1000.0, float(rpm))
        throt = max(0.0, min(100.0, float(throttle_pct))) / 100.0

        p_max, t_core, m_trap, tau_fric, deg = self.x

        # Physics transition models
        # Target trapped air mass based on throttle and displacement
        target_m_trap = 180.0 + 380.0 * throt * (rpm / 5000.0) ** 0.35
        # Target peak pressure proportional to trapped mass and boost
        target_p_max = 28.0 + (target_m_trap / 420.0) * 45.0 * (1.0 / max(0.8, deg))
        # Target core combustion temperature
        target_t_core = 1600.0 + 650.0 * (throt ** 0.5)
        # Friction torque increases with RPM^1.2 and degradation
        target_tau_fric = (8.5 + 7.5 * (rpm / 5000.0) ** 1.2) * deg

        # Continuous state decay towards operating point
        alpha_p = 1.0 - math.exp(-dt / 0.15)
        alpha_t = 1.0 - math.exp(-dt / 0.40)
        alpha_m = 1.0 - math.exp(-dt / 0.10)
        alpha_f = 1.0 - math.exp(-dt / 0.50)

        p_max_new = p_max + alpha_p * (target_p_max - p_max)
        t_core_new = t_core + alpha_t * (target_t_core - t_core)
        m_trap_new = m_trap + alpha_m * (target_m_trap - m_trap)
        tau_fric_new = tau_fric + alpha_f * (target_tau_fric - tau_fric)
        deg_new = deg + 0.000005 * dt  # slow nominal drift

        self.x = np.array([p_max_new, t_core_new, m_trap_new, tau_fric_new, deg_new])

        # State Jacobian matrix F = df/dx
        F = np.eye(self.state_dim)
        F[0, 0] = 1.0 - alpha_p
        F[1, 1] = 1.0 - alpha_t
        F[2, 2] = 1.0 - alpha_m
        F[3, 3] = 1.0 - alpha_f

        # Covariance extrapolation P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q * (dt / 0.1)

    def update(self, telem: TelemetryFrame) -> Dict[str, Any]:
        """
        Measurement-update step fusing multi-sensor frame.
        """
        # Form observation vector z
        z = np.array([
            telem.map_pa / 100000.0,
            telem.fuel_flow_lph,
            telem.cht_avg_c,
            telem.egt_avg_c,
            telem.oil_pressure_bar,
            telem.vibration_rms_g
        ])

        p_max, t_core, m_trap, tau_fric, deg = self.x

        # Non-linear observation function h(x) predicting measurements from states
        # 1. MAP (bar) relates to trapped mass: MAP ~ m_trap / (V_cyl * rho_norm)
        h_map = (m_trap / 380.0) * 1.05
        # 2. Fuel flow (L/h) = (m_trap * 4 * RPM/120) / AFR
        h_ff = (m_trap * 1e-6 * 4.0 * (telem.rpm / 120.0) / 14.2) * 3600.0 / 0.72
        # 3. CHT (°C) relates to heat rejection from peak pressure and core temperature
        h_cht = 35.0 + 0.025 * (t_core - 273.15) + 0.30 * p_max
        # 4. EGT (°C) is core temp minus expansion cooling
        h_egt = max(350.0, 120.0 + 0.35 * (t_core - 273.15) + 0.15 * p_max)
        # 5. Oil pressure drops slightly with degradation and high friction
        h_oil_p = max(1.0, 4.2 - 0.04 * (tau_fric - 12.0) - 0.5 * max(0.0, deg - 1.0))
        # 6. Vibration RMS scales with peak firing pressure and friction
        h_vib = max(0.4, 0.65 + 0.012 * p_max + 0.03 * (tau_fric - 10.0))

        h_x = np.array([h_map, h_ff, h_cht, h_egt, h_oil_p, h_vib])

        # Measurement residual (innovation) y = z - h(x)
        y = z - h_x

        # Measurement Jacobian H = dh/dx (evaluated at current state estimate)
        H = np.zeros((self.meas_dim, self.state_dim))
        H[0, 2] = 1.05 / 380.0                       # d(MAP)/d(m_trap)
        H[1, 2] = (1e-6 * 4.0 * (telem.rpm / 120.0) / 14.2) * 3600.0 / 0.72
        H[2, 0] = 0.30                               # d(CHT)/d(P_max)
        H[2, 1] = 0.025                              # d(CHT)/d(T_core)
        H[3, 0] = 0.15                               # d(EGT)/d(P_max)
        H[3, 1] = 0.35                               # d(EGT)/d(T_core)
        H[4, 3] = -0.04                              # d(OilP)/d(tau_fric)
        H[4, 4] = -0.50                              # d(OilP)/d(deg)
        H[5, 0] = 0.012                              # d(Vib)/d(P_max)
        H[5, 3] = 0.030                              # d(Vib)/d(tau_fric)
        H[4, 3] = -0.04                              # d(OilP)/d(tau_fric)
        H[4, 4] = -0.50                              # d(OilP)/d(deg)
        H[5, 0] = 0.012                              # d(Vib)/d(P_max)
        H[5, 3] = 0.030                              # d(Vib)/d(tau_fric)

        # Innovation covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Invert S with numerical stabilization
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        # Kalman gain K = P * H^T * S^-1
        K = self.P @ H.T @ S_inv

        # State update x = x + K * y
        self.x = self.x + K @ y

        # State bounds enforcement (physical constraints)
        self.x[0] = max(15.0, min(120.0, self.x[0]))     # P_max [15 - 120 bar]
        self.x[1] = max(1000.0, min(2800.0, self.x[1]))  # T_core [1000 - 2800 K]
        self.x[2] = max(50.0, min(700.0, self.x[2]))     # m_trapped [50 - 700 mg]
        self.x[3] = max(5.0, min(40.0, self.x[3]))       # Friction torque [5 - 40 Nm]
        self.x[4] = max(0.8, min(3.5, self.x[4]))        # Degradation [0.8 - 3.5]

        # Covariance update P = (I - K * H) * P (Joseph form for numerical symmetry)
        I_KH = np.eye(self.state_dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Normalized Mahalanobis distance D_M = sqrt( (y^T * S^-1 * y) / m )
        d_m_sq = float(y.T @ S_inv @ y)
        d_m_norm = math.sqrt(max(0.0, d_m_sq / self.meas_dim))
        self.mahalanobis_distance = round(d_m_norm, 2)

        # Standard deviations (confidence intervals 2*sigma)
        sigma = np.sqrt(np.maximum(1e-6, np.diag(self.P)))

        return {
            "estimated_p_max_bar": round(float(self.x[0]), 1),
            "p_max_confidence_pm_bar": round(float(2.0 * sigma[0]), 1),
            "estimated_t_core_k": round(float(self.x[1]), 0),
            "estimated_trapped_mass_mg": round(float(self.x[2]), 1),
            "estimated_friction_torque_nm": round(float(self.x[3]), 2),
            "degradation_index": round(float(self.x[4]), 3),
            "sensor_fusion_mahalanobis_distance": self.mahalanobis_distance,
            "sensor_consistency_status": "CONSISTENT" if self.mahalanobis_distance < 2.5 else "SENSOR_ANOMALY"
        }

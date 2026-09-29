"""
Engine Vibration and Kinematic Harmonic Signature Model
Synthesizes tri-axial vibration accelerations, order tracking (0.5x, 1x, 2x firing order),
RMS velocity, peak g-levels, crest factor, and spectral energy for MALE UAV aero piston engines.
"""

import math
import numpy as np
from typing import Dict, Any, List

class VibrationSignatureModel:
    def __init__(self):
        self.sampling_rate_hz = 1000.0  # Simulated vibration sensor sampling
        self.window_size = 256
        self.time_cursor = 0.0

    def compute(
        self,
        rpm: float,
        power_brake_kw: float,
        bmep_bar: float,
        misfire_active: bool = False,
        bearing_wear_severity: float = 0.0,       # 0.0 nominal to 1.0 damaged
        propeller_unbalance_severity: float = 0.0, # 0.0 balanced to 1.0 unbalance
        combustion_knock_severity: float = 0.0    # 0.0 no knock to 1.0 heavy detonation
    ) -> Dict[str, Any]:
        """
        Generate tri-axial vibration metrics and order harmonic components.
        """
        rpm = max(800.0, float(rpm))
        f0 = rpm / 60.0  # 1x fundamental rotational frequency (Hz)
        
        # Nominal amplitude scaling with engine speed and load
        load_factor = max(0.2, power_brake_kw / 100.0)
        base_rms_g = 0.85 + 0.95 * (rpm / 5800.0) ** 1.8 * load_factor

        # Order amplitudes:
        # 1. 0.5x (Sub-harmonic / Camshaft / Valve train / Misfire)
        amp_half_x = 0.08 + (0.95 * float(misfire_active))
        
        # 2. 1x (Propeller / Crankshaft mass unbalance)
        amp_one_x = (0.25 + 1.8 * propeller_unbalance_severity) * (rpm / 5000.0) ** 2
        
        # 3. 2x (4-cylinder firing frequency: 2 combustion events per rev)
        amp_two_x = (0.65 + 0.45 * (bmep_bar / 10.0))
        if misfire_active:
            # During misfire, 2x firing order is disrupted, transferring energy into 0.5x and 1x
            amp_two_x *= 0.55
            
        # 4. 4x (Higher combustion harmonics)
        amp_four_x = 0.18 * load_factor

        # 5. High-frequency broadband impact noise (Bearing cage/race defects, 1.2 kHz - 4 kHz)
        amp_high_freq = (0.12 + 1.4 * bearing_wear_severity + 2.2 * combustion_knock_severity)

        # Composite RMS acceleration (g)
        rms_g = math.sqrt(
            amp_half_x ** 2 +
            amp_one_x ** 2 +
            amp_two_x ** 2 +
            amp_four_x ** 2 +
            amp_high_freq ** 2
        ) * (base_rms_g / 1.2)

        # Peak g calculation with crest factor (spikes from misfire or knock)
        base_crest_factor = 2.4
        if misfire_active:
            base_crest_factor += 1.8
        if combustion_knock_severity > 0.1:
            base_crest_factor += 2.5 * combustion_knock_severity
        if bearing_wear_severity > 0.2:
            base_crest_factor += 1.2 * bearing_wear_severity

        peak_g = rms_g * base_crest_factor

        # Directional tri-axial breakdown (X: transverse rocking, Y: vertical, Z: axial/thrust)
        vib_x_rms_g = rms_g * 1.05
        vib_y_rms_g = rms_g * 0.88
        vib_z_rms_g = rms_g * 0.65

        # Synthetic order spectrum values for HMI / FFT strip display (in g)
        spectral_bins = [
            {"order": "0.5x (Cam/Misfire)", "freq_hz": round(f0 * 0.5, 1), "amplitude_g": round(amp_half_x, 3)},
            {"order": "1.0x (Shaft/Prop)", "freq_hz": round(f0 * 1.0, 1), "amplitude_g": round(amp_one_x, 3)},
            {"order": "2.0x (Firing Freq)", "freq_hz": round(f0 * 2.0, 1), "amplitude_g": round(amp_two_x, 3)},
            {"order": "4.0x (Harmonic)", "freq_hz": round(f0 * 4.0, 1), "amplitude_g": round(amp_four_x, 3)},
            {"order": "High-Freq/Acoustic", "freq_hz": 2400.0, "amplitude_g": round(amp_high_freq, 3)}
        ]

        return {
            "vibration_rms_g": float(rms_g),
            "vibration_peak_g": float(peak_g),
            "vibration_x_rms_g": float(vib_x_rms_g),
            "vibration_y_rms_g": float(vib_y_rms_g),
            "vibration_z_rms_g": float(vib_z_rms_g),
            "crest_factor": float(base_crest_factor),
            "firing_frequency_hz": float(f0 * 2.0),
            "amp_0_5x": float(amp_half_x),
            "amp_1_0x": float(amp_one_x),
            "amp_2_0x": float(amp_two_x),
            "amp_4_0x": float(amp_four_x),
            "amp_high_freq": float(amp_high_freq),
            "spectral_bins": spectral_bins,
            "knock_intensity": float(combustion_knock_severity)
        }

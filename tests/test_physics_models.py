"""
Rigorous Unit & Physical Validation Tests for Aero Piston Engine Physics Models
"""

import unittest
import math
from src.physics.atmosphere import AtmosphereModel
from src.physics.thermodynamic_cycle import ThermodynamicEngineModel
from src.physics.turbocharger import TurbochargerModel
from src.physics.thermal_network import ThermalNetworkModel
from src.physics.vibration_model import VibrationSignatureModel

class TestAeroEnginePhysics(unittest.TestCase):

    def setUp(self):
        self.atmos = AtmosphereModel()
        self.thermo = ThermodynamicEngineModel()
        self.turbo = TurbochargerModel()
        self.thermal = ThermalNetworkModel()
        self.vib = VibrationSignatureModel()

    def test_isa_sea_level(self):
        res = self.atmos.calculate(0.0)
        self.assertAlmostEqual(res["pressure_pa"], 101325.0, delta=10.0)
        self.assertAlmostEqual(res["temperature_c"], 15.0, delta=0.2)
        self.assertAlmostEqual(res["density_kg_m3"], 1.225, delta=0.01)

    def test_isa_high_altitude(self):
        # At 25,000 ft (7620 m), pressure should be approx 37-38 kPa, temp approx -34°C
        res = self.atmos.calculate(25000.0)
        self.assertLess(res["pressure_pa"], 40000.0)
        self.assertGreater(res["pressure_pa"], 35000.0)
        self.assertLess(res["temperature_c"], -30.0)

    def test_thermodynamic_power_bsfc(self):
        # Nominal cruise: 5000 RPM, MAP 101325 Pa (30 inHg)
        state = self.thermo.calculate_state(
            rpm=5000.0,
            manifold_pressure_pa=101325.0,
            manifold_temp_k=310.0
        )
        self.assertGreater(state["power_brake_hp"], 70.0)
        self.assertLess(state["power_brake_hp"], 150.0)
        # Brake thermal efficiency should be between 26% and 36%
        self.assertGreater(state["eta_brake_thermal"], 0.25)
        self.assertLess(state["eta_brake_thermal"], 0.40)
        # BSFC for modern 4-stroke aero piston engine ~ 240 - 330 g/kWh
        self.assertGreater(state["bsfc_g_kwh"], 230.0)
        self.assertLess(state["bsfc_g_kwh"], 350.0)

    def test_turbocharger_boost(self):
        # At 15,000 ft (ambient ~ 57 kPa), 100% throttle should boost MAP above ambient
        atmos = self.atmos.calculate(15000.0)
        turbo = self.turbo.compute(
            ambient_pressure_pa=atmos["pressure_pa"],
            ambient_temp_k=atmos["temperature_k"],
            throttle_pct=100.0,
            rpm=5500.0
        )
        self.assertGreater(turbo["map_pa"], atmos["pressure_pa"] * 1.5)
        self.assertGreater(turbo["wastegate_duty_pct"], 40.0)

    def test_thermal_transient_step(self):
        # Step thermal network under takeoff power
        out = self.thermal.step(
            dt_s=1.0,
            rpm=5500.0,
            power_indicated_kw=120.0,
            power_brake_kw=100.0,
            mass_fuelflow_kg_s=0.007,
            ambient_temp_c=15.0,
            airspeed_kts=85.0
        )
        self.assertIn("cht_1_c", out)
        self.assertIn("egt_1_c", out)
        self.assertGreater(out["cht_avg_c"], 70.0)
        self.assertLess(out["cht_avg_c"], 145.0)
        self.assertGreater(out["egt_avg_c"], 650.0)
        self.assertLess(out["egt_avg_c"], 880.0)

    def test_vibration_harmonics(self):
        vib = self.vib.compute(
            rpm=5000.0,
            power_brake_kw=80.0,
            bmep_bar=8.5,
            misfire_active=False
        )
        self.assertGreater(vib["vibration_rms_g"], 0.5)
        self.assertLess(vib["vibration_rms_g"], 2.5)
        self.assertAlmostEqual(vib["firing_frequency_hz"], (5000.0 / 60.0) * 2.0, delta=1.0)

if __name__ == "__main__":
    unittest.main()

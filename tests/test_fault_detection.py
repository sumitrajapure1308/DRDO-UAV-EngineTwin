"""
Rigorous Testing of Intelligent Fault Detection across all 8 Problem Statement Categories
"""

import unittest
import time
from src.core.state import TelemetryFrame, DigitalTwinEstimate
from src.diagnostics.fault_detector import IntelligentFaultDetector

class TestFaultDetectionAlgorithms(unittest.TestCase):

    def setUp(self):
        self.detector = IntelligentFaultDetector()

    def create_nominal_frames(self):
        now = time.time()
        telem = TelemetryFrame(
            timestamp_s=now,
            rpm=5000.0,
            throttle_pct=70.0,
            map_inhg=30.0,
            map_pa=101325.0,
            cht_1_c=95.0, cht_2_c=96.0, cht_3_c=98.0, cht_4_c=97.0,
            cht_avg_c=96.5, cht_spread_c=3.0,
            egt_1_c=760.0, egt_2_c=765.0, egt_3_c=770.0, egt_4_c=762.0,
            egt_avg_c=764.2, egt_spread_c=10.0,
            coolant_temp_c=85.0,
            oil_pressure_bar=4.2, oil_temp_c=90.0,
            fuel_flow_lph=24.5, fuel_pressure_bar=3.0,
            vibration_rms_g=1.4, crest_factor=2.4
        )
        twin = DigitalTwinEstimate(
            timestamp_s=now,
            twin_map_pa=101325.0, twin_map_inhg=30.0,
            twin_power_brake_kw=80.0, twin_power_brake_hp=107.0,
            twin_bmep_bar=8.5, twin_bsfc_g_kwh=275.0,
            twin_eta_thermal=0.31, twin_fuel_flow_lph=24.5,
            twin_cht_1_c=95.0, twin_cht_2_c=96.0, twin_cht_3_c=98.0, twin_cht_4_c=97.0,
            twin_egt_1_c=760.0, twin_egt_2_c=765.0, twin_egt_3_c=770.0, twin_egt_4_c=762.0,
            twin_coolant_temp_c=85.0, twin_oil_temp_c=90.0, twin_oil_pressure_bar=4.2,
            twin_vibration_rms_g=1.4,
            residual_map_pa=0.0,
            residual_cht_1_c=0.0, residual_cht_2_c=0.0, residual_cht_3_c=0.0, residual_cht_4_c=0.0,
            residual_egt_1_c=0.0, residual_egt_2_c=0.0, residual_egt_3_c=0.0, residual_egt_4_c=0.0,
            residual_fuel_flow_lph=0.0,
            residual_oil_pressure_bar=0.0, residual_oil_temp_c=0.0,
            residual_vibration_rms_g=0.0,
            composite_residual_norm=0.0
        )
        return telem, twin

    def test_nominal_condition_no_alerts(self):
        telem, twin = self.create_nominal_frames()
        alerts = self.detector.detect(telem, twin)
        self.assertEqual(len(alerts), 0)

    def test_misfire_detection_cylinder_3(self):
        telem, twin = self.create_nominal_frames()
        # Simulate misfire on Cylinder #3: EGT drops significantly, crest factor spikes
        telem.egt_3_c = 280.0
        telem.egt_avg_c = (telem.egt_1_c + telem.egt_2_c + 280.0 + telem.egt_4_c) / 4.0
        telem.crest_factor = 4.2
        telem.vibration_rms_g = 2.5
        twin.residual_egt_3_c = -490.0

        alerts = self.detector.detect(telem, twin)
        misfire_alerts = [a for a in alerts if "MISF" in a.fault_code]
        self.assertEqual(len(misfire_alerts), 1)
        self.assertEqual(misfire_alerts[0].affected_cylinder, 3)
        self.assertEqual(misfire_alerts[0].severity, "CRITICAL")

    def test_injector_abnormality_cylinder_2(self):
        telem, twin = self.create_nominal_frames()
        # Partial clog on Injector #2: runs lean, CHT rises, CHT spread increases
        telem.cht_2_c = 118.0
        telem.cht_spread_c = 23.0
        twin.residual_cht_2_c = 22.0

        alerts = self.detector.detect(telem, twin)
        inj_alerts = [a for a in alerts if "INJ" in a.fault_code]
        self.assertEqual(len(inj_alerts), 1)
        self.assertEqual(inj_alerts[0].affected_cylinder, 2)

    def test_cooling_degradation(self):
        telem, twin = self.create_nominal_frames()
        # Radiator blockage: global CHT rise + coolant temperature rise
        telem.coolant_temp_c = 104.0
        twin.residual_cht_1_c = 16.0
        twin.residual_cht_2_c = 17.0
        twin.residual_cht_3_c = 15.0
        twin.residual_cht_4_c = 16.0

        alerts = self.detector.detect(telem, twin)
        cool_alerts = [a for a in alerts if "COOL" in a.fault_code]
        self.assertEqual(len(cool_alerts), 1)
        self.assertEqual(cool_alerts[0].subsystem, "cooling_system")

    def test_lubrication_pressure_loss(self):
        telem, twin = self.create_nominal_frames()
        # Oil pressure drops to 1.1 bar
        telem.oil_pressure_bar = 1.1
        twin.residual_oil_pressure_bar = -3.1

        alerts = self.detector.detect(telem, twin)
        lube_alerts = [a for a in alerts if "LUBE" in a.fault_code]
        self.assertEqual(len(lube_alerts), 1)
        self.assertEqual(lube_alerts[0].severity, "CRITICAL")

    def test_sensor_drift_analytical_redundancy(self):
        telem, twin = self.create_nominal_frames()
        # CHT #4 reads +26°C high, but EGT, coolant, and oil remain perfectly nominal
        telem.cht_4_c = 123.0
        twin.residual_cht_4_c = 26.0
        twin.residual_egt_4_c = 1.0  # Combustion itself is normal!
        telem.coolant_temp_c = 85.0

        alerts = self.detector.detect(telem, twin)
        sens_alerts = [a for a in alerts if "SENS" in a.fault_code]
        self.assertEqual(len(sens_alerts), 1)
        self.assertEqual(sens_alerts[0].affected_cylinder, 4)

    def test_combustion_instability_knock(self):
        telem, twin = self.create_nominal_frames()
        telem.crest_factor = 3.9
        telem.vibration_rms_g = 2.4

        alerts = self.detector.detect(telem, twin)
        comb_alerts = [a for a in alerts if "COMB" in a.fault_code]
        self.assertEqual(len(comb_alerts), 1)

    def test_abnormal_mechanical_vibration(self):
        telem, twin = self.create_nominal_frames()
        telem.vibration_rms_g = 2.9  # Severe structural vibration
        telem.vibration_peak_g = 7.1

        alerts = self.detector.detect(telem, twin)
        vib_alerts = [a for a in alerts if "VIB" in a.fault_code]
        self.assertEqual(len(vib_alerts), 1)

if __name__ == "__main__":
    unittest.main()

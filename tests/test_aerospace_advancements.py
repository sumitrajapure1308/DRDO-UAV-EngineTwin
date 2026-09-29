"""
Unit and Integration Tests for Advanced Aerospace Modules:
1. Engine Performance Maps & Compressor Operating Boundaries
2. Extended Kalman Filter (EKF) Sensor Fusion State Estimator
3. Mission Reliability Enhancer & Tactical RTB Envelope Reconfiguration
4. 720-Degree Crank-Angle Resolved P-V Indicator Diagram
"""

import unittest
import numpy as np
from src.physics.engine_maps import EnginePerformanceMaps
from src.physics.thermodynamic_cycle import ThermodynamicEngineModel
from src.core.sensor_fusion import EngineSensorFusionEKF
from src.health.mission_reliability_enhancer import MissionReliabilityEnhancer
from src.core.state import TelemetryFrame, CompositeEngineHealth, SubsystemHealth

class TestAerospaceAdvancements(unittest.TestCase):

    def setUp(self):
        self.maps = EnginePerformanceMaps()
        self.thermo = ThermodynamicEngineModel()
        self.ekf = EngineSensorFusionEKF()
        self.enhancer = MissionReliabilityEnhancer()

    def test_engine_performance_maps_interpolation(self):
        # Test volumetric efficiency interpolation
        eta_v = self.maps.lookup_volumetric_efficiency(rpm=4800, map_bar=1.15)
        self.assertGreater(eta_v, 0.85)
        self.assertLess(eta_v, 0.98)

        # Test BSFC island lookup
        bsfc = self.maps.lookup_bsfc(rpm=4800, bmep_bar=9.5)
        self.assertGreater(bsfc, 230.0)
        self.assertLess(bsfc, 255.0)

        # Test compressor aerodynamic point
        comp = self.maps.evaluate_compressor_operating_point(corrected_flow_kg_s=0.08, pressure_ratio=1.30)
        self.assertEqual(comp["aerodynamic_status"], "STABLE")
        self.assertGreater(comp["surge_margin_pct"], 10.0)

        # Test surge detection
        surge_comp = self.maps.evaluate_compressor_operating_point(corrected_flow_kg_s=0.02, pressure_ratio=1.65)
        self.assertTrue(surge_comp["is_surging"])
        self.assertEqual(surge_comp["aerodynamic_status"], "SURGE")

    def test_indicator_pv_cycle_simulation(self):
        pv = self.thermo.calculate_indicator_pv_cycle(map_pa=135000.0, rpm=5500.0, combustion_efficiency=0.98)
        self.assertIn("volume_cm3", pv)
        self.assertIn("pressure_bar", pv)
        self.assertGreater(pv["peak_cylinder_pressure_bar"], 50.0)
        self.assertLess(pv["peak_cylinder_pressure_bar"], 110.0)
        self.assertEqual(len(pv["volume_cm3"]), len(pv["pressure_bar"]))
        self.assertGreater(pv["indicated_work_joules_per_cyl"], 100.0)

    def test_ekf_sensor_fusion_convergence(self):
        sample_frame = TelemetryFrame(
            rpm=5200.0,
            throttle_pct=75.0,
            map_pa=125000.0,
            fuel_flow_lph=26.5,
            cht_avg_c=108.0,
            egt_avg_c=745.0,
            oil_pressure_bar=4.1,
            vibration_rms_g=1.4
        )

        # Run 5 prediction-update cycles to simulate filter convergence
        for _ in range(5):
            self.ekf.predict(rpm=sample_frame.rpm, throttle_pct=sample_frame.throttle_pct, dt_s=0.1)
            ekf_res = self.ekf.update(sample_frame)

        self.assertGreater(ekf_res["estimated_p_max_bar"], 30.0)
        self.assertGreater(ekf_res["estimated_t_core_k"], 1500.0)
        self.assertLess(ekf_res["estimated_t_core_k"], 2600.0)
        self.assertEqual(ekf_res["sensor_consistency_status"], "CONSISTENT")
        self.assertLess(ekf_res["sensor_fusion_mahalanobis_distance"], 3.8)

    def test_mission_reliability_enhancer_nominal_and_derating(self):
        telem_nom = TelemetryFrame(
            rpm=5000.0,
            altitude_ft=12000.0,
            airspeed_kts=110.0,
            oil_pressure_bar=4.2,
            cht_1_c=98.0, cht_2_c=99.0, cht_3_c=97.0, cht_4_c=98.0,
            fuel_flow_lph=22.0
        )
        health_nom = CompositeEngineHealth(overall_health_index=0.95)
        
        # Nominal evaluation
        nom_env = self.enhancer.evaluate(telem_nom, health_nom, fuel_remaining_liters=45.0)
        self.assertEqual(nom_env["mission_directive"], "GO_FULL_MISSION")
        self.assertEqual(nom_env["max_safe_throttle_pct"], 100.0)
        self.assertGreater(nom_env["total_reachable_range_nm"], 200.0)

        # Critical overheating & oil pressure drop -> Emergency Derating
        telem_fail = TelemetryFrame(
            rpm=5000.0,
            altitude_ft=12000.0,
            airspeed_kts=110.0,
            oil_pressure_bar=1.6,  # severe oil drop
            cht_1_c=144.0, cht_2_c=146.0, cht_3_c=143.0, cht_4_c=145.0,  # overheat
            fuel_flow_lph=24.0
        )
        health_fail = CompositeEngineHealth(overall_health_index=0.48)
        fail_env = self.enhancer.evaluate(telem_fail, health_fail, fuel_remaining_liters=45.0)

        self.assertIn("ABORT", fail_env["mission_directive"])
        self.assertLessEqual(fail_env["max_safe_throttle_pct"], 55.0)
        self.assertGreater(len(fail_env["derate_reasons"]), 1)

    def test_sensor_redundancy_virtual_synthesis(self):
        from src.health.sensor_redundancy import AnalyticalSensorRedundancy
        from src.core.digital_twin_engine import DigitalTwinSynchronizer
        redundancy = AnalyticalSensorRedundancy()
        twin_engine = DigitalTwinSynchronizer()

        telem_nom = TelemetryFrame(
            cht_1_c=98.0,
            cht_2_c=98.5,
            cht_3_c=99.0,
            cht_4_c=97.0,
            map_inhg=29.92,
            rpm=5000.0
        )
        twin = twin_engine.synchronize(telem_nom)

        telem_broken_cht2 = TelemetryFrame(
            cht_1_c=98.0,
            cht_2_c=310.0, # open-circuit thermocouple failure
            cht_3_c=99.0,
            cht_4_c=97.0,
            map_inhg=29.92,
            rpm=5000.0
        )

        # Run 6 cycles to exceed debounce threshold
        for _ in range(6):
            res = redundancy.evaluate_and_synthesize(telem_broken_cht2, twin)

        self.assertTrue(res["virtual_substitutions_active"])
        self.assertIn("cht_2", res["active_channels"])
        synth_data = res["substitutions"]["cht_2"]
        self.assertEqual(synth_data["failure_mode"], "OPEN_CIRCUIT")
        # Synthesized value should be near nominal ~89.7°C, NOT the broken 310°C
        self.assertLess(synth_data["synthesized_virtual"], 120.0)
        self.assertGreater(synth_data["synthesized_virtual"], 75.0)

    def test_offline_trained_ai_model_loading(self):
        from src.ai_ml.hybrid_detector import HybridPhysicsAIDetector
        detector = HybridPhysicsAIDetector()
        self.assertTrue(detector.is_model_trained)
        self.assertAlmostEqual(detector.calibrated_recon_threshold, 0.2096, places=3)

if __name__ == "__main__":
    unittest.main()

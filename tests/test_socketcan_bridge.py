"""
Unit Test for SocketCAN and HIL Communication Bridge
"""

import unittest
import time
from src.core.socketcan_bridge import SocketCANBridge
from src.core.bus import CANFrame

class TestSocketCANBridge(unittest.TestCase):

    def setUp(self):
        self.bridge = SocketCANBridge(interface="vcan0", udp_port=5556)
        self.received_frames = []

        def on_frame(frame: CANFrame):
            self.received_frames.append(frame)

        self.bridge.start(frame_callback=on_frame)

    def tearDown(self):
        self.bridge.stop()

    def test_bridge_transmission_and_reception(self):
        # Transmit a test frame (0x200: RPM and MAP)
        test_frame = CANFrame(can_id=0x200, data=b"\x13\x88\x27\x10\x96\x03\xe8")
        self.bridge.send_frame(test_frame)

        # Allow brief time for socket loopback
        time.sleep(0.1)

        self.assertGreater(self.bridge.tx_count, 0)
        self.assertGreaterEqual(self.bridge.rx_count, 1)
        self.assertEqual(self.received_frames[0].can_id, 0x200)

    def test_hil_can_protocol_roundtrip(self):
        from src.core.bus import AeroCANProtocol
        from src.core.state import TelemetryFrame
        
        sample_frame = TelemetryFrame(
            timestamp_s=time.time(),
            mission_time_s=120.0,
            rpm=5500.0,
            throttle_pct=85.0,
            altitude_ft=15000.0,
            airspeed_kts=110.0,
            ambient_temp_c=-14.0,
            ambient_pressure_pa=57182.0,
            map_pa=130000.0,
            map_inhg=38.4,
            cht_1_c=115.0, cht_2_c=118.0, cht_3_c=116.0, cht_4_c=117.0,
            cht_avg_c=116.5, cht_spread_c=3.0,
            egt_1_c=745.0, egt_2_c=750.0, egt_3_c=742.0, egt_4_c=748.0,
            egt_avg_c=746.2, egt_spread_c=8.0,
            oil_pressure_bar=4.2,
            oil_temp_c=98.0,
            coolant_temp_c=85.0,
            fuel_flow_lph=28.5,
            fuel_pressure_bar=3.0,
            air_fuel_ratio=13.8,
            vibration_rms_g=1.45,
            vibration_peak_g=3.2,
            crest_factor=2.2,
            bus_voltage_v=28.2,
            alternator_current_a=24.0,
            battery_current_a=2.0
        )
        can_frames = AeroCANProtocol.encode_telemetry_to_can(sample_frame)
        self.assertEqual(len(can_frames), 7)
        expected_ids = {0x200, 0x201, 0x202, 0x203, 0x204, 0x205, 0x206}
        self.assertEqual({f.can_id for f in can_frames}, expected_ids)

        for f in can_frames:
            self.bridge.send_frame(f)

        time.sleep(0.1)
        self.assertGreaterEqual(self.bridge.tx_count, 7)
        self.assertGreaterEqual(self.bridge.rx_count, 7)

if __name__ == "__main__":
    unittest.main()

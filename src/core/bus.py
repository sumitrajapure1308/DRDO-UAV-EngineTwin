"""
Telemetry Ingestion Bus and CAN / SocketCAN Protocol Interface
Provides high-throughput async event dispatch and aerospace CAN frame (0x200-0x208)
encoding/decoding for MALE UAV ECU/FADEC telemetry ingestion.
"""

import asyncio
import struct
import time
from typing import Callable, List, Dict, Any, Optional
from .state import TelemetryFrame

class CANFrame:
    """Standard CAN 2.0B / UAVCAN telemetry frame representation."""
    def __init__(self, can_id: int, data: bytes, timestamp_s: float = None):
        self.can_id = can_id
        self.data = data
        self.timestamp_s = timestamp_s or time.time()

    def __repr__(self):
        return f"<CANFrame ID=0x{self.can_id:03X} DLC={len(self.data)} DATA={self.data.hex()}>"


class TelemetryBus:
    """
    Central telemetry dispatcher supporting subscriber listeners and virtual CAN bus bridge.
    """
    def __init__(self):
        self._subscribers: List[Callable[[TelemetryFrame], None]] = []
        self._async_subscribers: List[asyncio.Queue] = []
        self.packet_count = 0
        self.byte_count = 0
        self.last_frame: Optional[TelemetryFrame] = None

    def subscribe(self, callback: Callable[[TelemetryFrame], None]):
        self._subscribers.append(callback)

    def subscribe_queue(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=100)
        self._async_subscribers.append(q)
        return q

    def unsubscribe_queue(self, q: asyncio.Queue):
        if q in self._async_subscribers:
            self._async_subscribers.remove(q)

    def publish(self, frame: TelemetryFrame):
        self.packet_count += 1
        self.last_frame = frame
        
        # Dispatch to synchronous callbacks
        for callback in self._subscribers:
            try:
                callback(frame)
            except Exception as e:
                pass

        # Dispatch to async queues (dropping oldest if buffer full)
        for q in list(self._async_subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass


class AeroCANProtocol:
    """
    Encodes/Decodes aero piston engine FADEC telemetry into CAN bus frames.
    Message ID Mapping (AeroCAN Standard):
      0x200: RPM (uint16), MAP_hPa (uint16), Throttle_pct (uint8), Alt_m (uint16)
      0x201: CHT1, CHT2, CHT3, CHT4 (4x uint16, °C * 10)
      0x202: EGT1, EGT2, EGT3, EGT4 (4x uint16, °C * 10)
      0x203: Oil_Press_kPa (uint16), Oil_Temp_C (int16), Coolant_Temp_C (int16)
      0x204: FuelFlow_g_s (uint16), FuelPressure_kPa (uint16), AFR (uint16 * 100)
      0x205: Vib_RMS_mG (uint16), Vib_Peak_mG (uint16), CrestFactor (uint16 * 100)
      0x206: BusVoltage_mV (uint16), AltCurrent_cA (uint16), BatCurrent_cA (int16)
    """

    @staticmethod
    def encode_telemetry_to_can(frame: TelemetryFrame) -> List[CANFrame]:
        frames = []
        now = frame.timestamp_s

        # 0x200
        rpm_val = int(max(0, min(65535, frame.rpm)))
        map_hpa = int(max(0, min(65535, frame.map_pa / 100.0)))
        throttle_byte = int(max(0, min(255, frame.throttle_pct * 2.55)))
        alt_m = int(max(0, min(65535, frame.altitude_ft * 0.3048)))
        d0 = struct.pack(">HHBH", rpm_val, map_hpa, throttle_byte, alt_m)[:8]
        frames.append(CANFrame(0x200, d0, now))

        # 0x201 (CHTs)
        c1 = int(max(0, min(65535, frame.cht_1_c * 10)))
        c2 = int(max(0, min(65535, frame.cht_2_c * 10)))
        c3 = int(max(0, min(65535, frame.cht_3_c * 10)))
        c4 = int(max(0, min(65535, frame.cht_4_c * 10)))
        frames.append(CANFrame(0x201, struct.pack(">HHHH", c1, c2, c3, c4), now))

        # 0x202 (EGTs)
        e1 = int(max(0, min(65535, frame.egt_1_c * 10)))
        e2 = int(max(0, min(65535, frame.egt_2_c * 10)))
        e3 = int(max(0, min(65535, frame.egt_3_c * 10)))
        e4 = int(max(0, min(65535, frame.egt_4_c * 10)))
        frames.append(CANFrame(0x202, struct.pack(">HHHH", e1, e2, e3, e4), now))

        # 0x203 (Oil & Coolant)
        oil_p_kpa = int(max(0, min(65535, frame.oil_pressure_bar * 100.0)))
        oil_t = int(frame.oil_temp_c * 10)
        cool_t = int(frame.coolant_temp_c * 10)
        frames.append(CANFrame(0x203, struct.pack(">Hhh", oil_p_kpa, oil_t, cool_t), now))

        # 0x204 (Fuel system)
        ff_c = int(frame.fuel_flow_lph * 10)
        fp_kpa = int(frame.fuel_pressure_bar * 100)
        afr_val = int(frame.air_fuel_ratio * 100)
        frames.append(CANFrame(0x204, struct.pack(">HHH", ff_c, fp_kpa, afr_val), now))

        # 0x205 (Vibration)
        vib_rms_mg = int(max(0, min(65535, frame.vibration_rms_g * 1000.0)))
        vib_peak_mg = int(max(0, min(65535, frame.vibration_peak_g * 1000.0)))
        crest_factor_x100 = int(max(0, min(65535, frame.crest_factor * 100.0)))
        frames.append(CANFrame(0x205, struct.pack(">HHH", vib_rms_mg, vib_peak_mg, crest_factor_x100), now))

        # 0x206 (Electrical Bus)
        bus_v_mv = int(max(0, min(65535, frame.bus_voltage_v * 1000.0)))
        alt_i_ca = int(max(0, min(65535, frame.alternator_current_a * 100.0)))
        bat_i_ca = int(max(-32768, min(32767, frame.battery_current_a * 100.0)))
        frames.append(CANFrame(0x206, struct.pack(">HHh", bus_v_mv, alt_i_ca, bat_i_ca), now))

        return frames

    @staticmethod
    def decode_can_frame(frame: CANFrame, telem_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decodes incoming raw CAN frame (0x200 - 0x206) and updates telemetry dictionary.
        """
        if frame.can_id == 0x200 and len(frame.data) >= 7:
            rpm_val, map_hpa, throttle_byte, alt_m = struct.unpack(">HHBH", frame.data[:7])
            telem_dict["rpm"] = float(rpm_val)
            telem_dict["map_pa"] = float(map_hpa * 100.0)
            telem_dict["map_inhg"] = float((map_hpa * 100.0) / 3386.389)
            telem_dict["throttle_pct"] = float(throttle_byte / 2.55)
            telem_dict["altitude_ft"] = float(alt_m / 0.3048)
        elif frame.can_id == 0x201 and len(frame.data) >= 8:
            c1, c2, c3, c4 = struct.unpack(">HHHH", frame.data[:8])
            telem_dict["cht_1_c"] = float(c1 / 10.0)
            telem_dict["cht_2_c"] = float(c2 / 10.0)
            telem_dict["cht_3_c"] = float(c3 / 10.0)
            telem_dict["cht_4_c"] = float(c4 / 10.0)
            telem_dict["cht_avg_c"] = float((c1 + c2 + c3 + c4) / 40.0)
            telem_dict["cht_spread_c"] = float((max(c1, c2, c3, c4) - min(c1, c2, c3, c4)) / 10.0)
        elif frame.can_id == 0x202 and len(frame.data) >= 8:
            e1, e2, e3, e4 = struct.unpack(">HHHH", frame.data[:8])
            telem_dict["egt_1_c"] = float(e1 / 10.0)
            telem_dict["egt_2_c"] = float(e2 / 10.0)
            telem_dict["egt_3_c"] = float(e3 / 10.0)
            telem_dict["egt_4_c"] = float(e4 / 10.0)
            telem_dict["egt_avg_c"] = float((e1 + e2 + e3 + e4) / 40.0)
            telem_dict["egt_spread_c"] = float((max(e1, e2, e3, e4) - min(e1, e2, e3, e4)) / 10.0)
        elif frame.can_id == 0x203 and len(frame.data) >= 6:
            oil_p_kpa, oil_t, cool_t = struct.unpack(">Hhh", frame.data[:6])
            telem_dict["oil_pressure_bar"] = float(oil_p_kpa / 100.0)
            telem_dict["oil_temp_c"] = float(oil_t / 10.0)
            telem_dict["coolant_temp_c"] = float(cool_t / 10.0)
        elif frame.can_id == 0x204 and len(frame.data) >= 6:
            ff_c, fp_kpa, afr_val = struct.unpack(">HHH", frame.data[:6])
            telem_dict["fuel_flow_lph"] = float(ff_c / 10.0)
            telem_dict["fuel_pressure_bar"] = float(fp_kpa / 100.0)
            telem_dict["air_fuel_ratio"] = float(afr_val / 100.0)
        elif frame.can_id == 0x205 and len(frame.data) >= 6:
            vib_rms_mg, vib_peak_mg, crest_factor_x100 = struct.unpack(">HHH", frame.data[:6])
            telem_dict["vibration_rms_g"] = float(vib_rms_mg / 1000.0)
            telem_dict["vibration_peak_g"] = float(vib_peak_mg / 1000.0)
            telem_dict["crest_factor"] = float(crest_factor_x100 / 100.0)
        elif frame.can_id == 0x206 and len(frame.data) >= 6:
            bus_v_mv, alt_i_ca, bat_i_ca = struct.unpack(">HHh", frame.data[:6])
            telem_dict["bus_voltage_v"] = float(bus_v_mv / 1000.0)
            telem_dict["alternator_current_a"] = float(alt_i_ca / 100.0)
            telem_dict["battery_current_a"] = float(bat_i_ca / 100.0)
        return telem_dict

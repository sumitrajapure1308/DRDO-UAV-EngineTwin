"""
SocketCAN and Hardware-In-The-Loop (HIL) Communication Bridge
Supports native Linux SocketCAN (AF_CAN) and cross-platform UDP CAN frame emulation
for direct hardware interfacing with UAV engine test cells and FADEC ECUs.
"""

import socket
import struct
import time
import sys
import threading
from typing import Callable, Optional, Dict, Any
from .bus import CANFrame, AeroCANProtocol, TelemetryBus
from .state import TelemetryFrame

class SocketCANBridge:
    """
    Bidirectional CAN bridge for Hardware-In-The-Loop (HIL) engine test benches.
    Supports native SocketCAN on Linux and UDP CAN emulation on Windows.
    """
    def __init__(self, interface: str = "vcan0", udp_port: int = 5555, bus: Optional[TelemetryBus] = None):
        self.interface = interface
        self.udp_port = udp_port
        self.bus = bus or TelemetryBus()
        self.running = False
        self.sock = None
        self.is_native_can = hasattr(socket, "AF_CAN")
        self.rx_count = 0
        self.tx_count = 0

    def start(self, frame_callback: Optional[Callable[[CANFrame], None]] = None):
        """Start listening for incoming CAN frames in a background thread."""
        self.running = True
        if self.is_native_can:
            # Native Linux SocketCAN
            try:
                self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
                self.sock.bind((self.interface,))
                print(f"[*] Bound to native Linux SocketCAN interface: {self.interface}")
            except Exception as e:
                print(f"[!] Warning: Native SocketCAN bind failed ({e}), falling back to UDP CAN emulator.")
                self.is_native_can = False

        if not self.is_native_can:
            # Cross-platform UDP CAN Socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("0.0.0.0", self.udp_port))
            print(f"[*] Bound to Cross-Platform UDP CAN Bridge on port {self.udp_port}")

        self.rx_thread = threading.Thread(target=self._rx_loop, args=(frame_callback,), daemon=True)
        self.rx_thread.start()

    def _rx_loop(self, callback: Optional[Callable[[CANFrame], None]]):
        while self.running:
            try:
                if self.is_native_can:
                    data, _ = self.sock.recvfrom(16)
                    can_id, can_dlc, can_data = self._unpack_native_can(data)
                else:
                    data, _ = self.sock.recvfrom(64)
                    # UDP CAN format: 4-byte can_id (big-endian), 1-byte dlc, data bytes
                    can_id, can_dlc = struct.unpack(">IB", data[:5])
                    can_data = data[5:5 + can_dlc]

                frame = CANFrame(can_id=can_id, data=can_data, timestamp_s=time.time())
                self.rx_count += 1

                if callback:
                    callback(frame)
            except Exception:
                if not self.running:
                    break

    def send_frame(self, frame: CANFrame, dest_ip: str = "127.0.0.1"):
        """Transmit CAN frame to the bus or test rig."""
        if not self.sock:
            return
        self.tx_count += 1
        if self.is_native_can:
            # Native Linux CAN frame struct: uint32 can_id, uint8 can_dlc, uint8 pad, uint8 res0, uint8 res1, uint8 data[8]
            can_pkt = struct.pack("=IB3x8s", frame.can_id, len(frame.data), frame.data.ljust(8, b"\x00"))
            self.sock.send(can_pkt)
        else:
            payload = struct.pack(">IB", frame.can_id, len(frame.data)) + frame.data
            self.sock.sendto(payload, (dest_ip, self.udp_port))

    def _unpack_native_can(self, data: bytes):
        can_id, can_dlc, data_bytes = struct.unpack("=IB3x8s", data)
        return can_id & 0x1FFFFFFF, can_dlc, data_bytes[:can_dlc]

    def stop(self):
        self.running = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass

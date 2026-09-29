"""
DRDO MALE UAV Aero Piston Engine Digital Twin
Hardware-In-The-Loop (HIL) CAN Bus Simulation & Avionics Test-Bench Driver

Simulates an onboard FADEC / Engine Control Unit (ECU) transmitting 
aerospace CAN 2.0B / UAVCAN telemetry frames into the Digital Twin over 
native Linux SocketCAN (AF_CAN) or Cross-Platform UDP CAN (Port 5555/5556).

Usage:
  python run_hil_simulation.py --duration 10 --rate 10
  python run_hil_simulation.py --fault misfire --cyl 2
  python run_hil_simulation.py --fault oil_loss
"""

import sys
import time
import argparse
import socket
import struct
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.bus import AeroCANProtocol, CANFrame
from src.simulation.mission_simulator import MissionSimulator
from src.simulation.fault_injector import FaultInjector

def run_hil_test(
    duration_s: float = 15.0,
    rate_hz: float = 10.0,
    dest_ip: str = "127.0.0.1",
    udp_port: int = 5555,
    fault_type: str = "none",
    fault_cyl: int = 2,
    fault_time_s: float = 4.0
):
    print("=" * 78)
    print("   DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)")
    print("   Hardware-In-The-Loop (HIL) Avionics CAN Test Rig")
    print(f"   Target: {dest_ip}:{udp_port} | Rate: {rate_hz} Hz | Duration: {duration_s} s")
    if fault_type != "none":
        print(f"   Scheduled In-Flight Fault: {fault_type.upper()} at T+{fault_time_s}s (Cyl #{fault_cyl})")
    print("=" * 78)

    # Instantiate full physics simulator and fault injector
    injector = FaultInjector()
    sim = MissionSimulator(fault_injector=injector)

    # UDP socket for cross-platform CAN transport
    tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    dt = 1.0 / rate_hz
    start_time = time.time()
    t_sim = 0.0
    frame_counter = 0
    fault_injected = False

    print(f"[*] Starting HIL transmission loop at {rate_hz} Hz...")
    print(f"{'TIME':<8} | {'RPM':<6} | {'MAP (inHg)':<10} | {'CHT2 (°C)':<9} | {'EGT2 (°C)':<9} | {'OIL P (bar)':<11} | {'TX FRAMES':<9} | {'STATUS'}")
    print("-" * 88)

    try:
        while t_sim < duration_s:
            cycle_start = time.time()
            t_sim = cycle_start - start_time

            # Handle fault injection trigger
            if fault_type != "none" and t_sim >= fault_time_s and not fault_injected:
                injector.inject_fault(fault_type, severity=0.75, target_cylinder=fault_cyl)
                fault_injected = True

            # Step full high-fidelity physics model
            frame = sim.step(dt_s=dt)

            # Encode into standard CAN 2.0B / UAVCAN frames
            can_frames = AeroCANProtocol.encode_telemetry_to_can(frame)

            # Transmit frames over UDP CAN socket
            for f in can_frames:
                udp_payload = struct.pack(">IB", f.can_id, len(f.data)) + f.data
                tx_sock.sendto(udp_payload, (dest_ip, udp_port))
                frame_counter += 1

            # Format status
            status_str = "NOMINAL"
            if fault_injected:
                status_str = f"FAULT: {fault_type.upper()} (CYL {fault_cyl})"

            # Print telemetry summary at 1 Hz
            if frame_counter % (len(can_frames) * int(rate_hz)) < len(can_frames):
                print(f"T+{t_sim:04.1f}s | {frame.rpm:6.0f} | {frame.map_inhg:10.1f} | {frame.cht_2_c:9.1f} | {frame.egt_2_c:9.1f} | {frame.oil_pressure_bar:11.2f} | {frame_counter:<9} | {status_str}")

            # Pace to exact rate
            elapsed = time.time() - cycle_start
            sleep_time = max(0.001, dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[!] User interrupted HIL simulation.")
    finally:
        tx_sock.close()
        print("\n" + "=" * 78)
        print(f"[+] HIL SIMULATION COMPLETE: Transmitted {frame_counter} CAN frames over {t_sim:.1f} s.")
        print(f"    Effective Bus Rate: {frame_counter / max(0.1, t_sim):.1f} frames/sec")
        print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRDO MALE UAV Aero Engine HIL CAN Simulator")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration in seconds (default: 10.0)")
    parser.add_argument("--rate", type=float, default=10.0, help="CAN transmission rate in Hz (default: 10.0)")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Target UDP IP (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Target UDP Port (default: 5555)")
    parser.add_argument("--fault", type=str, default="none", choices=["none", "misfire", "injector_abnormal", "cooling_degradation", "lubrication_issue", "sensor_drift", "combustion_instability", "overheating_trend", "abnormal_vibration"], help="Fault injection type")
    parser.add_argument("--cyl", type=int, default=2, choices=[1, 2, 3, 4], help="Target cylinder for fault (default: 2)")
    parser.add_argument("--fault-time", type=float, default=3.0, help="Time in seconds to inject fault (default: 3.0)")

    args = parser.parse_args()
    run_hil_test(
        duration_s=args.duration,
        rate_hz=args.rate,
        dest_ip=args.ip,
        udp_port=args.port,
        fault_type=args.fault,
        fault_cyl=args.cyl,
        fault_time_s=args.fault_time
    )

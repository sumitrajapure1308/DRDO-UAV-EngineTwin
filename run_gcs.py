"""
DRDO MALE UAV Aero Piston Engine Digital Twin System - Master Launcher
Launches the real-time telemetry simulator, physics engine, AI diagnostics, and GCS Web Dashboard.
"""

import sys
import os
import argparse
import uvicorn
from pathlib import Path

# Ensure the project root is in python path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main():
    parser = argparse.ArgumentParser(description="DRDO MALE UAV Aero Piston Engine Digital Twin System (SIH26054)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    args = parser.parse_args()

    print("=" * 78)
    print("   DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)")
    print("   AI-Enabled Real-Time Digital Twin System for Aero Piston Engines")
    print("   Problem Statement ID: SIH26054 | Department of Defence R&D")
    print("=" * 78)
    print(f"[*] Initializing Digital Twin Core Framework in {ROOT_DIR}...")
    print(f"[*] Ground Control Station (GCS) Dashboard URL: http://{args.host}:{args.port}/dashboard")
    print(f"[*] Telemetry WebSocket Endpoint: ws://{args.host}:{args.port}/ws/telemetry")
    print(f"[*] REST API Status Endpoint: http://{args.host}:{args.port}/api/status")
    print("=" * 78)

    uvicorn.run("src.api.server:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()

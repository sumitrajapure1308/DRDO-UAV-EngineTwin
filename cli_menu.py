"""
DRDO MALE UAV Aero Piston Engine Digital Twin (SIH26054)
Interactive Evaluation & Demonstration Console
"""

import sys
import os
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent.resolve()

def print_header():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 78)
    print("   DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)")
    print("   AI-Enabled Real-Time Digital Twin System for Aero Piston Engines")
    print("   Problem Statement: SIH26054 (MALE UAV Propulsion Health & Prognostics)")
    print("=" * 78)

def main_menu():
    while True:
        print_header()
        print("\n  OPERATIONAL VERIFICATION & DEMONSTRATION MENU:")
        print("  " + "-" * 50)
        print("  [1] Run Full Master Demonstration (All 6 Phases)")
        print("  [2] Run Automated Test Discovery Suite (37 Aerospace Tests)")
        print("  [3] Run Edge AI Embedded Inference Benchmark")
        print("  [4] Run Hardware-In-The-Loop (HIL) CAN Bus Simulation")
        print("  [5] Run Live WebSocket & REST API Integration Verification")
        print("  [6] Ingest & Generate Debrief Report for All Flight Sorties")
        print("  [7] Launch Ground Control Station (GCS) Dashboard Server")
        print("  [8] Run Offline Physics-Residual ML Model Training & Calibration")
        print("  [9] Open Documentation & Technical Specifications")
        print("  [0] Exit")
        print("  " + "-" * 50)
        
        choice = input("\n  Select Option [0-9]: ").strip()
        
        if choice == "1":
            print("\n[>] Executing Master Demonstration...")
            subprocess.run([sys.executable, str(ROOT_DIR / "demonstrate_all.py")])
            input("\nPress Enter to return to menu...")
        elif choice == "2":
            print("\n[>] Running 35 Unit & Integration Tests...")
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"], cwd=str(ROOT_DIR))
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            print("\n[>] Running Edge AI Inference Benchmark...")
            subprocess.run([sys.executable, "-c", "from src.ai_ml.edge_benchmark import run_edge_ai_benchmark; run_edge_ai_benchmark(500)"], cwd=str(ROOT_DIR))
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            print("\n[>] Launching Hardware-In-The-Loop (HIL) CAN Simulator (15s run)...")
            subprocess.run([sys.executable, str(ROOT_DIR / "run_hil_simulation.py"), "--duration", "15", "--rate", "10", "--fault", "misfire", "--cyl", "2"])
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            print("\n[>] Verifying Live WebSocket & REST endpoints...")
            subprocess.run([sys.executable, str(ROOT_DIR / "tests" / "live_verify.py")])
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            print("\n[>] Processing all reference flight sortie logs...")
            recordings_dir = ROOT_DIR / "recordings"
            for f in recordings_dir.glob("*_sample.jsonl"):
                print(f"\n[*] Processing: {f.name}")
                subprocess.run([sys.executable, "-m", "src.cli", str(f), "--output-dir", "reports"], cwd=str(ROOT_DIR))
            input("\nPress Enter to return to menu...")
        elif choice == "7":
            print("\n[>] Starting GCS Dashboard at http://127.0.0.1:8000/dashboard...")
            if os.name == "nt":
                os.system(f'start "" http://127.0.0.1:8000/dashboard')
            subprocess.run([sys.executable, str(ROOT_DIR / "run_gcs.py"), "--port", "8000", "--host", "0.0.0.0"])
        elif choice == "8":
            print("\n[>] Running Physics-Residual ML Model Training & Calibration...")
            subprocess.run([sys.executable, str(ROOT_DIR / "src" / "ai_ml" / "train_models.py")], cwd=str(ROOT_DIR))
            input("\nPress Enter to return to menu...")
        elif choice == "9":
            spec_file = ROOT_DIR / "docs" / "DRDO_SYSTEM_SPECIFICATION.md"
            print(f"\n[>] Technical Specification available at: {spec_file}")
            readme_file = ROOT_DIR / "README.md"
            print(f"[>] Comprehensive README available at: {readme_file}")
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("\nExiting DRDO Digital Twin Console. Jai Hind!\n")
            sys.exit(0)

if __name__ == "__main__":
    main_menu()

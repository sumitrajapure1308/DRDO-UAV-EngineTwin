@echo off
REM ==============================================================================
REM DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)
REM Aero Piston Engine Digital Twin - Ground Control Station (GCS) Launcher
REM Problem Statement: SIH26054
REM ==============================================================================

echo [>] Starting DRDO MALE UAV Digital Twin GCS Server on port 8000...
start "" http://127.0.0.1:8000/dashboard
python run_gcs.py --port 8000 --host 0.0.0.0
pause

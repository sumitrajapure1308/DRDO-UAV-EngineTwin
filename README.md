# DRDO | Department of Defence R&D
## Problem Statement ID: SIH26054
### AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs

---

## 1. Executive Summary

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs) perform long-duration intelligence, surveillance, reconnaissance (ISR), maritime patrol, and tactical defense missions. The propulsion powerplant—typically a high-performance 4-cylinder turbocharged aero piston engine (Rotax 914/915 iS class)—is single-fault critical to mission success. Conventional threshold-based monitoring systems are reactive, alerting operators only after severe physical abnormalities or damage have already occurred.

This repository presents an **indigenous, physics-informed, AI-enabled Real-Time Digital Twin Core Framework** developed specifically to meet all requirements of **DRDO Problem Statement 26054 (SIH26054)**. The system provides:
1. **Real-Time Digital Twin Virtual Engine Model** synchronized with live telemetry and FADEC CAN data at 10 Hz / 20 Hz.
2. **Multi-Subsystem Health Monitoring & Indexing** across all 8 required engine parameters.
3. **Intelligent Predictive Diagnostics** detecting all 8 required failure modes before redline threshold breach.
4. **Hybrid AI/ML Layer** combining physics residuals with deep autoencoder reconstruction, isolation forest anomaly scoring, and Explainable AI (XAI) root-cause attribution.
5. **Stochastic Remaining Useful Life (RUL) Prognostics** using a Wiener process degradation model with $P_{10}, P_{50}, P_{90}$ probabilistic confidence intervals.
6. **Multi-Mission Environmental Simulation & Replay Engine** supporting High Altitude (25,000 ft), Hot Desert (+46°C), Rapid Throttle Tactical Maneuvers, Maritime Patrol, and in-flight dynamic fault injection.
7. **Defense-Grade Ground Control Station (GCS) Visualization Dashboard** with an interactive 4-cylinder engine cutaway, live thermal heatmap, harmonic vibration order spectrum, annunciator alerts, and automated DRDO mission health reports.

---

## 2. System Architecture

```
                          [ MALE UAV FLIGHT SENSORS / FADEC ECU ]
                                            │
                                  CAN Bus / SocketCAN
                                  (0x200 - 0x206 IDs)
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          TELEMETRY INGESTION & EVENT BUS                               │
│                         (AeroCANProtocol / TelemetryBus)                               │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DIGITAL TWIN CORE SYNCHRONIZER                                  │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │   ISA Atmosphere      │  │  4-Stroke Otto/Dual    │  │  Turbocharger & Charge    │  │
│  │   Lapse Rate Model    │  │  Thermodynamic Cycle   │  │  Air Intercooler Physics  │  │
│  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
│  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
│  │   4-Cylinder Thermal  │  │  Tri-Axial Vibration   │  │  State Observer & Dynamic │  │
│  │   Network & Oil Model │  │  Kinematic Orders      │  │  Physics Residual Tracker │  │
│  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┼───────────────────────────┐
               ▼                            ▼                           ▼
┌─────────────────────────────┐ ┌───────────────────────┐ ┌──────────────────────────┐
│   HEALTH MONITORING SYSTEM  │ │   DIAGNOSTIC ENGINE   │ │       AI/ML LAYER          │
│ • Thermal CHT Health        │ │ • Misfire Detection   │ │ • Residual Deep Autoencoder│
│ • Exhaust Gas Path Health   │ │ • Injector Clog/Leak  │ │ • Isolation Forest Anomaly │
│ • Fuel Injection Health     │ │ • Cooling Degradation │ │ • Wiener RUL Prognostics   │
│ • Lubrication Circuit Health│ │ • Lubrication Deficit │ │   (P10, P50, P90 Bounds)   │
│ • Turbo Boost Health        │ │ • Sensor Drift (Analyt│ │ • Explainable AI (XAI)     │
│ • Coolant Loop Health       │ │   Redundancy)         │ │   Feature Attribution      │
│ • Combustion Stability      │ │ • Knock & Instability │ │ • Prescriptive Advisory    │
│ • Electrical Bus Health     │ │ • Overheating Trend   │ │   Work Order Generator     │
│ • Composite EHI Synthesis   │ │ • Mechanical Vibration│ └──────────────────────────┘
└─────────────────────────────┘ └───────────────────────┘               │
               │                            │                           │
               └────────────────────────────┼───────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               DEFENSE GROUND CONTROL STATION (GCS) WEB HMI & REST API                  │
│ • Live 10 Hz Telemetry Streaming (WebSockets)                                          │
│ • 4-Cylinder Interactive Engine Cutaway Schematic with Thermal Heatmap                │
│ • Real-Time Dual-Needle Instrumentation & Strip Chart Recorder                        │
│ • In-Flight Dynamic Fault Injector & Manual Throttle Slew                             │
│ • Automated DRDO Mission Health & Debrief Report Generator (HTML / JSON / Printable)   │
│ • Flight Data Recorder (FDR) Black-Box Replay with Timeline Scrubbing                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Subsystem Health Monitoring (Requirement B)

The framework continuously monitors and indexes 8 critical engine subsystems:
1. **RPM & Kinematics:** Tracks engine speed, overspeed limits, and rotational acceleration.
2. **Cylinder Head Temperature (CHT 1..4):** Assesses absolute thermal limits (Caution 135°C, Redline 145°C) and cylinder-to-cylinder thermal spread $\Delta \text{CHT}_{spread}$.
3. **Exhaust Gas Temperature (EGT 1..4):** Evaluates combustion efficiency, individual cylinder AFR divergence, and cylinder spread $\Delta \text{EGT}_{spread}$.
4. **Lubrication System:** Evaluates hydrodynamic oil film pressure vs. oil temperature envelope, identifying bearing clearance wear and oil dilution.
5. **Fuel Flow & Pressure:** Checks fuel pressure stability (2.8 - 3.2 bar) and fuel mass flow residual against indicated thermodynamic demand.
6. **Turbocharger & Boost Control:** Monitors compressor pressure ratio, manifold absolute pressure (MAP), and electronic wastegate duty cycle.
7. **Combustion Stability & Vibration:** Measures tri-axial RMS vibration acceleration, peak acceleration, crest factor, and harmonic order tracking (0.5x, 1x, 2x, HF knock band).
8. **Electrical Bus & Alternator:** Evaluates 28V DC bus regulation, alternator output current, battery charge/discharge, and AC voltage ripple (alternator rectifier diode breakdown).

**Composite Engine Health Index (EHI):**
$$\text{EHI} = \sum_{i=1}^{8} w_i \cdot \text{HI}_i \quad \text{where } \text{EHI} \in [0.0, 1.0]$$

---

## 4. Intelligent Predictive Diagnostics (Requirement C)

The system replaces reactive threshold alarms with predictive physics-residual diagnostics for all 8 required fault categories:

| Fault Mode | Diagnostic Signature & Detection Methodology | ATA Code | Severity |
| :--- | :--- | :--- | :--- |
| **Misfire Conditions** | Localized EGT quenching ($>130^\circ\text{C}$ drop below siblings/twin) + sharp subharmonic 0.5x vibration order spike + crest factor $>3.2$. Isolates misfiring cylinder (1..4). | ATA72-MISF-01 | CRITICAL |
| **Injector Abnormalities** | Cylinder-to-cylinder fuel delivery imbalance. Localized CHT elevation ($>12^\circ\text{C}$ residual) or EGT spread $>38^\circ\text{C}$ without dead-cylinder collapse. Isolates defective injector. | ATA73-INJ-02 | WARNING |
| **Cooling Degradation** | Uniform global CHT rise across all 4 cylinder heads accompanied by coolant temperature rise ($>96^\circ\text{C}$) despite adequate ram airspeed. | ATA75-COOL-01 | WARNING |
| **Lubrication Issues** | Oil pressure dropping below hydrodynamic threshold ($<2.0$ bar at cruise, or $<0.9$ bar redline) or severe pressure deficit relative to digital twin model. | ATA79-LUBE-01 | CRITICAL |
| **Sensor Drift / Failure** | Analytical physics redundancy: probe deviates by $>18^\circ\text{C}$ while cross-coupled thermodynamic variables (EGT, coolant, oil) confirm nominal engine behavior. | ATA77-SENS-01 | CAUTION |
| **Combustion Instability** | High-frequency detonation acoustic energy, vibration crest factor $>3.6$, and cyclic dispersion under high MAP / high intake charge temp. | ATA72-COMB-01 | WARNING |
| **Overheating Trends** | Predictive thermal derivative extrapolation: $\text{CHT}_{projected}(t + 180s) = \text{CHT}(t) + \frac{d\text{CHT}}{dt} \cdot 180$. Triggers warning before redline is reached. | ATA75-OHT-02 | CAUTION |
| **Abnormal Vibration** | Structural RMS vibration $>2.6g$ or elevated 1.0x order (propeller/shaft unbalance) or elevated high-frequency band (bearing raceway impact). | ATA72-VIB-02 | WARNING |

---

## 5. AI/ML Layer & RUL Prognostics (Requirement D)

### Hybrid Physics-Informed Anomaly Detection
- Extracts normalized physics residual vector:
  $$\mathbf{r} = \left[ \frac{\Delta \text{MAP}}{\sigma_{\text{MAP}}}, \frac{\Delta \text{CHT}_{1..4}}{\sigma_{\text{CHT}}}, \frac{\Delta \text{EGT}_{1..4}}{\sigma_{\text{EGT}}}, \frac{\Delta \dot{m}_{fuel}}{\sigma_{ff}}, \frac{\Delta P_{oil}}{\sigma_{oil\_p}}, \frac{\Delta T_{oil}}{\sigma_{oil\_t}}, \frac{\Delta \text{Vib}}{\sigma_{vib}} \right]$$
- **Deep PyTorch Autoencoder:** 13-input, 32-16-4 latent bottleneck network trained to reconstruct nominal physics residuals. High reconstruction loss flags non-linear physical degradation.
- **Isolation Forest:** Multi-variable density scoring on residual projections.
- **Explainable AI (XAI):** Decomposes anomaly scores into normalized Shapley-like percentage contributions for operators and engineers.

### Remaining Useful Life (RUL) Prognostics
- Stochastic degradation modeling via a Wiener process with drift $\mu(t)$ and diffusion $\sigma$:
  $$dD(t) = \mu_{deg}(t) dt + \sigma_{deg} dW(t)$$
- Drift rate dynamically escalates under thermal, lubrication, or combustion stress.
- First-hitting-time (FHT) median estimate ($P_{50}$ in flight hours) to caution (0.65 EHI) and critical overhaul (0.35 EHI) limits:
  $$\text{RUL}_{P50} = \frac{\text{EHI}(t) - \text{EHI}_{crit}}{\mu_{deg}(t)}$$
- Probabilistic bounds ($P_{10}$ pessimistic, $P_{90}$ optimistic) computed via Inverse Gaussian distribution quantiles.
- Evaluates mission completion probability $P(\text{RUL} > T_{\text{mission}})$ for the remaining sortie.

---

## 6. Simulation & Mission Profiles (Requirement E)

The simulator includes realistic MALE UAV flight mission profiles:
1. **High-Altitude ISR Surveillance:** Climb to 24,000 ft, extreme low ambient density, turbocharger operating at high wastegate pressure ratio, cold soak ($-30^\circ\text{C}$).
2. **Hot-Weather Desert Operation:** Ground taxi and loiter in $+46^\circ\text{C}$ ambient desert conditions with severe thermal stress on cooling and lubrication circuits.
3. **Rapid Throttle & Tactical Maneuvering:** Aggressive combat orbits, wave-offs, rapid power slams ($15\% \to 100\%$), transient turbo lag, and thermal shock.
4. **Maritime Long Endurance Patrol:** 24-hour low-altitude oceanic surveillance with thermal equilibrium and steady cruise consumption.
5. **Controlled Fault Evaluation Sortie:** Baseline nominal takeoff transitioning into staged fault injections to validate detection algorithms and RUL re-estimation.

---

## 7. Defense Ground Control Station (GCS) Dashboard (Requirement F)

- **Aero Propulsion Instruments:** Dual-needle/arc dials for RPM, MAP (inHg/bar), Brake Power (HP/kW), Torque (N·m), Fuel Flow, Fuel Pressure, Oil Pressure & Temperature, Coolant Temperature, and BSFC.
- **Interactive 4-Cylinder Engine Schematic:** Animated horizontally-opposed 4-cylinder reciprocating cutaway showing real-time firing sequence ($1 \to 4 \to 3 \to 2$), piston stroke motion, and dynamic cylinder head thermal color mapping.
- **Digital Twin Synchronizer Table:** Side-by-side comparison of measured telemetry vs. virtual twin model with real-time discrepancy residuals and composite Euclidean norm.
- **Annunciator & Advisory Panel:** Live MIL-STD color-coded alert feed with ATA chapter classification and prescriptive ground maintenance work orders.
- **Vibration Order Waterfall / FFT:** Order spectrum bars for 0.5x, 1.0x, 2.0x, and high-frequency knock bands.
- **Real-Time Strip Chart Recorder:** Dynamic 60-second rolling Canvas telemetry chart.
- **One-Click DRDO Mission Report:** Compiles a comprehensive engineering debrief with statistics, thermal extremes, fault logs, and work orders, exportable to printable HTML and JSON.

---

## 8. Verification & Execution Instructions

### Prerequisites
- Python 3.10+ (Verified on Python 3.11.0)
- Dependencies installed from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

### Step 1: Run the Complete Automated Test Suite
Execute the 27 unit and integration tests across all modules:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
*Expected Output:*
```
Ran 27 tests in 0.507s
OK
```

### Step 2: Launch the Digital Twin GCS Server
Launch the master launcher:
```bash
python run_gcs.py --port 8000
```
*Endpoints launched:*
- **GCS Web Dashboard:** `http://127.0.0.1:8000/dashboard`
- **Telemetry WebSocket:** `ws://127.0.0.1:8000/ws/telemetry`
- **REST Status:** `http://127.0.0.1:8000/api/status`

### Step 3: Run Live WebSocket & Fault Injection Verification
While the server is running, execute the live integration verifier:
```bash
python tests/live_verify.py
```
*Validates in real-time:*
- REST API handshake (`ONLINE`)
- 10 Hz WebSocket telemetry streaming
- In-flight fault injection over WebSocket (Cylinder #2 Injector Clog)
- Verification of alert trigger and Explainable AI (XAI) feature attribution
- Fault clearing and DRDO mission report generation

---

## 9. Compliance Matrix Against DRDO Problem Statement SIH26054

| Requirement from SIH26054 | Implementation Status | Implementation Details |
| :--- | :---: | :--- |
| **A. Digital Twin Core Framework** | **FULLY IMPLEMENTED** | Real-time virtual engine model synchronized via `DigitalTwinSynchronizer`, physics residual tracking, modular architecture, CAN bus ingestion. |
| **B. Health Monitoring System** | **FULLY IMPLEMENTED** | Continuous monitoring and health indexing for RPM, CHT (1..4), EGT (1..4), Oil P/T, Fuel Flow/P, Vibration harmonics, Battery/Alternator, Injection timing. |
| **C. Fault Detection & Predictive Analytics** | **FULLY IMPLEMENTED** | Multi-subsystem predictive detector for all 8 fault modes: misfire, injector abnormal, cooling degradation, oil issues, sensor drift, knock, overheating trends, vibration patterns. |
| **D. AI/ML Layer** | **FULLY IMPLEMENTED** | Hybrid physics-residual Deep Autoencoder, Isolation Forest anomaly detection, Wiener process RUL prognostics ($P_{10}, P_{50}, P_{90}$), Explainable AI (XAI). |
| **E. Simulation & Replay Capability** | **FULLY IMPLEMENTED** | Multi-mission flight simulator (High Altitude, Desert, Rapid Throttle, Maritime, Fault Evaluation), real-time fault injector, black-box FDR recorder and replay. |
| **F. Visualization Dashboard** | **FULLY IMPLEMENTED** | Defense GCS HMI with 4-cylinder animated engine cutaway, thermal gradient mapping, telemetry dials, strip charts, annunciator panel, and automated debrief reports. |

---

## 10. Hardware-In-The-Loop (HIL) & Deployment Roadmap

1. **Phase 1 (Completed in Software Prototype):** High-fidelity digital twin core, physics models, AI diagnostics, and GCS visualization.
2. **Phase 2 (HIL Test Rig Integration):** Connect `AeroCANProtocol` via SocketCAN to physical PEAK-System / Kvaser CAN transceivers interfacing directly with engine test bench FADEC ECUs.
3. **Phase 3 (Edge Embedded Deployment):** Compile `PhysicsResidualAutoencoder` using ONNX Runtime / TensorRT for deployment on onboard UAV edge computers (Raspberry Pi CM4 or Jetson Orin Nano).
4. **Phase 4 (Fleet-Level Ground Station):** Deploy the multi-threaded GCS server to defense ground stations for concurrent health monitoring across entire MALE UAV squadrons.

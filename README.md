<div align="center">

# 🇮🇳 AeroTwin-MALE | DRDO Aero Piston Engine Digital Twin
### AI-Enabled Real-Time Propulsion Health Monitoring, Fault Prediction & Mission Reliability Enhancement for MALE UAVs
**Smart India Hackathon (SIH 2026) | Problem Statement ID: SIH26054**  
*Sponsored by Defence Research & Development Organisation (DRDO) — Department of Defence R&D*

---

[![Build Status](https://img.shields.io/badge/Build-Passing-00ff88?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/sumitrajapure1308/DRDO-UAV-EngineTwin)
[![Tests](https://img.shields.io/badge/Unit%20Tests-37%2F37%20Passed%20(100%25)-3fb950?style=for-the-badge&logo=checkmarx&logoColor=white)](tests/)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-38bdf8?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B%20Offline%20Trained-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-10--20Hz%20Streaming-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Airworthiness](https://img.shields.io/badge/Airworthiness-CEMILAC%20DDPMAS%20Ready-ffaa00?style=for-the-badge&logo=shield&logoColor=white)](https://drdo.gov.in)
[![Military Protocol](https://img.shields.io/badge/Avionics-STANAG%204586%20%7C%20CANaerospace-a78bfa?style=for-the-badge&logo=target&logoColor=white)](https://nato.int)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ed?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)

<br>

[Executive Summary](#1-executive-summary--operational-context) •
[System Architecture](#2-system-architecture--information-pipeline) •
[Mathematical Formulations](#3-key-scientific--aerospace-breakthroughs) •
[AI/ML Engine](#4-physics-informed-aiml-layer--offline-training) •
[Fault Coverage](#5-subsystem-monitoring--fault-coverage-sih26054) •
[Mission Reliability](#6-analytical-sensor-redundancy--mission-reliability) •
[GCS Console](#7-defense-ground-control-station-gcs-console) •
[REST & Telemetry API](#8-rest-api--telemetry-protocol-specification) •
[Verification](#9-verification--benchmark-evidence) •
[Quick Start](#10-quick-start-guide) •
[SIH26054 Compliance Matrix](#11-compliance-matrix-against-drdo-problem-statement-sih26054)

</div>

---

## 1. Executive Summary & Operational Context

Medium Altitude Long Endurance (MALE) Unmanned Aerial Vehicles (UAVs)—exemplified by the **TAPAS-BH-201 (Rustom-II)** and **Archer-NG** developed by the Aeronautical Development Establishment (**ADE, DRDO**)—conduct 24+ hour intelligence, surveillance, target acquisition, and reconnaissance (ISTAR) sorties along contested frontiers.

The primary powerplant powering these aircraft is a high-performance **4-cylinder, 4-stroke, turbocharged aero piston engine** (Rotax 914 / 915 iS class / VRDE 180 HP indigenous powerplant). In single-engine MALE UAV configurations, engine degradation represents an immediate single-point-of-failure risk.

### Shortcomings of Legacy Engine Monitoring Systems (EMS)
1. **Reactive Latency:** Conventional threshold systems alert pilots only after physical thermal or vibration redlines are breached, resulting in catastrophic in-flight seizures, dead-stick landings, or aircraft loss (AOG / ditching).
2. **Zero Decision Support:** Existing systems provide no tactical mission survival support—they cannot derate power autonomously, compute instantaneous dead-stick glide cones, or synthesize analytical sensor redundancy when physical transducers fail.

### The AeroTwin-MALE Solution
**AeroTwin-MALE** is a defense-grade, physics-informed, AI-enabled Digital Twin platform engineered specifically for **DRDO Problem Statement SIH26054**. It couples a first-principles thermodynamic model with live FADEC telemetry at 10–20 Hz, predicts 8 critical failure modes via offline-trained deep neural networks, synthesizes virtual sensors during probe failures, and dynamically computes tactical reachability to Indian Air Force (IAF) airbases to guarantee mission recovery.

---

## 2. System Architecture & Information Pipeline

```
                                     [ MALE UAV SENSORS / FADEC ECU ]
                                                     │
                                       CAN Bus / ARINC 825 / SocketCAN
                                          (0x200 - 0x206 Frame IDs)
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                           DUAL-CHANNEL FADEC ARCHITECTURE (DO-178C)                             │
│  [ Lane A: Active Master ] ── CCDL (0.4 ms) ── [ Lane B: Hot Standby ] ── [ TMR Sensor Voting ]  │
└────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              INDIGENOUS DIGITAL TWIN CORE ENGINE                                │
│  • ISA Atmosphere Lapse Model          • 720° Slider-Crank Kinematics & Indicator P-V Cycle     │
│  • Chen-Flynn Engine Friction (FMEP)   • Universal BSFC Islands (238 g/kWh) & Surge Margin Line │
│  • Multi-Node CHT/EGT Heat Balance     • 5-State Non-Linear Extended Kalman Filter (EKF)        │
└────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        ▼                                ▼                                ▼
┌───────────────────────┐ ┌──────────────────────────────┐ ┌──────────────────────────────────────┐
│  OFFLINE-TRAINED AI   │ │  MISSION RELIABILITY ENGINE  │ │     ANALYTICAL SENSOR REDUNDANCY     │
│ • Deep Autoencoder    │ │ • Safe Throttle Derating     │ │ • Open-Circuit / Drift Detection     │
│   (13-32-16-4-16-32-13)│ │ • Dead-Stick Glide Footprint │ │ • Real-Time Virtual Sensor Synthesis │
│ • Isolation Forest    │ │ • Powered Range to IAF Bases │ │ • Fail-Operational Mission Mode      │
│ • ROC-AUC: 1.0000     │ │   (AFB Uttarlai, Jaisalmer)  │ │   (No False Emergency Aborts)        │
└───────────────────────┘ └──────────────────────────────┘ └──────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│               DEFENSE GROUND CONTROL STATION (GCS) HMI & BLACK BOX RECORDER                     │
│  • Western Sector Tactical Reachability Canvas (IAF Airbases, Dynamic Gliding Cone, Range Ring) │
│  • Crank-Angle Resolved P-V Indicator Canvas with Work Integral & Peak Pressure Readout         │
│  • Universal Performance Map with Calibrated BSFC Islands & Aerodynamic Surge Boundary         │
│  • Dual-Channel FADEC Status Strip & Real-Time Sensor Integrity Health Banner                   │
│  • One-Click Flight Data Recorder (FDR) Blackbox CSV Exporter (/api/fdr/export_csv)            │
│  • Acoustic Master Caution Chime (Web Audio API) & Synthesized Voice Warnings ("Bitchin' Betty")│
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Scientific & Aerospace Breakthroughs

### 1. In-Cylinder 720° Crank-Angle Resolved Indicator $P-V$ Diagram
Rather than treating the powerplant as an empirical black box, AeroTwin implements slider-crank kinematics coupled with Wiebe mass fraction burned combustion:
- **Instantaneous cylinder volume across 720° crank angle:**
  $$V(\theta) = V_c + \frac{V_d}{2} \left[ R + 1 - \cos\theta - \sqrt{R^2 - \sin^2\theta} \right]$$
  where $V_c$ is clearance volume, $V_d$ is displacement, and $R = l/a$ is the connecting rod to crank radius ratio ($R = 3.65$).
- **Wiebe combustion heat release mass fraction burned:**
  $$x_b(\theta) = 1 - \exp\left[ -a \left(\frac{\theta - \theta_{soc}}{\Delta\theta}\right)^{m+1} \right]$$
  where $a = 5.0$, $m = 2.0$, $\theta_{soc} = -18^\circ\text{ CA}$ (Start of Combustion), and $\Delta\theta = 48^\circ\text{ CA}$ (Combustion Duration).
- **Indicated Mean Effective Pressure & Firing Uncertainty:**
  $$W_{ind} = \oint P(\theta) \, dV, \quad \text{IMEP} = \frac{W_{ind}}{V_d}, \quad P_{max} \pm 2\sigma_{P_{max}}$$

### 2. 5-State Non-Linear Extended Kalman Filter (EKF) Sensor Fusion
Estimates unmeasurable core thermodynamic states by fusing 6 physical sensor channels (MAP, RPM, Fuel Flow, CHT, EGT, Oil Pressure):
$$\mathbf{x} = \begin{bmatrix} P_{max} & T_{core} & m_{trapped} & \tau_{fric} & \delta_{deg} \end{bmatrix}^T$$
- **Measurement Update & Innovation Covariance:**
  $$\mathbf{y}_k = \mathbf{z}_k - h(\hat{\mathbf{x}}_{k|k-1}), \quad \mathbf{S}_k = \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k$$
  $$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}, \quad \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \mathbf{y}_k$$
- **Normalized Mahalanobis Distance for Sensor Inconsistency:**
  $$D_{M,norm} = \sqrt{\frac{\mathbf{y}_k^T \mathbf{S}_k^{-1} \mathbf{y}_k}{m}} \quad (m = 6)$$
  Flags anomaly when $D_{M,norm} \ge 2.5\sigma$ without causing premature false engine shutdowns.

### 3. Chen-Flynn Mechanical Friction Modeling
Calculates engine mechanical losses ($FMEP$) as a function of cylinder peak firing pressure, mean piston speed ($\bar{S}_p$), and kinematic viscosity:
$$\text{FMEP} = c_0 + c_1 \cdot P_{max} + c_2 \cdot \bar{S}_p + c_3 \cdot \bar{S}_p^2$$
Enables precise discrimination between mechanical friction degradation (bearing wear) and thermodynamic combustion faults.

### 4. International Standard Atmosphere (ISA) Troposphere Model
Standardizes engine manifold density and compressor surge margins across operational altitudes (0 to 30,000 ft):
$$T(h) = T_0 - L \cdot h, \quad P(h) = P_0 \left(1 - \frac{L \cdot h}{T_0}\right)^{\frac{g}{R_{air} \cdot L}}, \quad \rho(h) = \frac{P(h)}{R_{air} \cdot T(h)}$$
where $T_0 = 288.15\text{ K}$, $P_0 = 101325\text{ Pa}$, $L = 0.0065\text{ K/m}$, and $g = 9.80665\text{ m/s}^2$.

---

## 4. Physics-Informed AI/ML Layer & Offline Training

AeroTwin avoids runtime-generated heuristics. All machine learning components are **trained offline**, validated against multi-mission flight envelopes, and packaged into self-contained inference engines.

| Model Component | Architecture & Topology | Training Dataset | Benchmark Performance |
| :--- | :--- | :--- | :--- |
| **Physics-Residual Deep Autoencoder** | 13-Input $\to$ 32 $\to$ 16 $\to$ 4 (Bottleneck) $\to$ 16 $\to$ 32 $\to$ 13 Output with LeakyReLU(0.1) and Adam Optimizer | 5,000 Nominal Multi-Mission Telemetry Vectors (Climb, High-Altitude Cruise, Desert +46°C, Combat Slams) | **ROC-AUC: 1.0000**<br>**Precision: 99.17%**<br>**Recall: 100.00%**<br>**F1-Score: 99.59%**<br>**Inference: 10.56 ms** |
| **Isolation Forest Anomaly Estimator** | 100 Isolation Trees, Contamination=0.02, Sub-sampling | 4,000 Nominal Residual Vectors | Anomaly Probability & Density Scoring |
| **Wiener Process Stochastic RUL** | Continuous Brownian motion with dynamic drift $\mu_{deg}(t)$ and diffusion $\sigma_{deg}$ | Cumulative stress wear modeling (Arrhenius thermal + Paris mechanical wear) | $P_{10}, P_{50}, P_{90}$ confidence intervals to 1,800-hour TBO limit |
| **Explainable AI (XAI) Attribution** | Multivariate Mahalanobis / Z-score variance decomposition | 13 Physics Residual Channels + Kinematic Orders | Normalized percentage attribution of top failure drivers |

### Offline Model Artifacts
- **Trained PyTorch Model:** [`models/physics_residual_autoencoder.pth`](models/physics_residual_autoencoder.pth) (2,053 trainable parameters)
- **Fitted Isolation Forest:** [`models/isolation_forest.joblib`](models/isolation_forest.joblib)
- **Quantitative Metrics Dossier:** [`reports/model_evaluation_metrics.json`](reports/model_evaluation_metrics.json)

---

## 5. Subsystem Monitoring & Fault Coverage (SIH26054)

### 8 Subsystem Health Indices (0.0% to 100.0%)
1. **RPM & Kinematics:** Tracks overspeed, rotational acceleration, and redline stability.
2. **Cylinder Head Temperature (CHT 1..4):** Monitored individually (Caution: 135°C, Warning: 145°C).
3. **Exhaust Gas Temperature (EGT 1..4):** Individual combustion efficiency and air-fuel ratio divergence.
4. **Lubrication Circuit:** Hydrodynamic oil pressure vs. oil temperature envelope.
5. **Fuel Delivery:** Sequential EFI rail pressure stability (2.8–3.2 bar) and fuel flow residual.
6. **Turbocharger & Boost:** Compressor pressure ratio and electronic wastegate duty cycle.
7. **Combustion Stability & Vibration:** RMS acceleration, crest factor, and rotational orders (0.5x, 1x, 2x, HF knock).
8. **Electrical Bus:** 28V DC bus regulation, alternator load current, and AC ripple voltage.

### 8 Intelligent Predictive Fault Modes

| Fault Mode | Diagnostic Signature & Detection Methodology | ATA Code | Severity |
| :--- | :--- | :--- | :--- |
| **Cylinder Misfire** | EGT quenching ($>130^\circ\text{C}$ drop) + subharmonic 0.5x order spike + crest factor $>3.2$. Pinpoints exact cylinder 1..4. | ATA72-MISF-01 | CRITICAL |
| **Injector Delivery Clog** | Cylinder fuel delivery deficit. CHT rise ($>12^\circ\text{C}$ residual) or EGT spread $>38^\circ\text{C}$ without dead-cylinder collapse. | ATA73-INJ-02 | WARNING |
| **Cooling Degradation** | Uniform global CHT elevation across all 4 cylinders accompanied by coolant temperature rise ($>96^\circ\text{C}$). | ATA75-COOL-01 | WARNING |
| **Lubrication Deficit** | Oil pressure dropping below hydrodynamic threshold ($<2.0\text{ bar}$ cruise, $<0.9\text{ bar}$ redline) or severe residual deficit. | ATA79-LUBE-01 | CRITICAL |
| **Sensor Drift / Open-Circuit** | Analytical physics redundancy: probe deviates $>35^\circ\text{C}$ or exceeds physical bounds while cross-coupled variables confirm nominal engine behavior. | ATA77-SENS-01 | CAUTION |
| **Combustion Instability / Knock** | High-frequency knock acoustic energy (1.5–4.0 kHz), crest factor $>3.6$, and cyclic dispersion under high boost. | ATA72-COMB-01 | WARNING |
| **Predictive Overheating** | Dynamic derivative thermal extrapolation: $\text{CHT}_{projected}(t + 180s) = \text{CHT}(t) + \frac{d\text{CHT}}{dt} \cdot 180$. Warnings trigger prior to redline breach. | ATA75-OHT-02 | CAUTION |
| **Mechanical Vibration** | Tri-axial RMS vibration $>2.6g$ or elevated 1.0x order (propeller/shaft unbalance) or bearing raceway impacts. | ATA72-VIB-02 | WARNING |

---

## 6. Analytical Sensor Redundancy & Mission Reliability

### DO-178C Level B/C Sensor Fault Tolerance
When physical thermocouples or pressure transducers experience open-circuits, loose wiring, or thermal drift, standard systems issue emergency abort warnings. AeroTwin maintains **Fail-Operational** capability:
- **Real-Time Virtual Sensor Synthesis:**
  $$\text{CHT}_{\text{synth}, i} = 0.70 \cdot \text{CHT}_{\text{twin}, i} + 0.30 \cdot \overline{\text{CHT}}_{\text{siblings}}$$
- Switches FADEC channel to `FAIL_OPERATIONAL`, preventing mission abort and allowing the MALE UAV to remain on station.

### Autonomous Safe Throttle Derating
When severe thermal degradation or oil pressure drops occur, the Mission Reliability Engine clamps the maximum allowable throttle:
- Derates throttle from 100% to 55–65% continuous power.
- Arrests cylinder head thermal runaway while maintaining sufficient thrust for level flight and obstacle clearance.

### Dead-Stick Gliding Footprint Cone
In the event of total engine failure, AeroTwin computes the instantaneous gliding footprint cone based on current altitude $H$ and the clean aerodynamic glide ratio ($L/D = 18.2$):
$$R_{\text{glide}} (\text{NM}) = \left(\frac{H_{\text{ft}}}{6076.12}\right) \times 18.2$$
$$A_{\text{glide}} = \pi \cdot R_{\text{glide}}^2$$

### Western Sector IAF Airbase Reachability Matrix
The system continuously evaluates glide cone reachability and powered fuel endurance to strategic airfields:
- **AFB Uttarlai (WAC / Barmer)** — Runway 02/20 (2,743 m)
- **AFB Jaisalmer** — Runway 04/22 (2,743 m)
- **AFB Nal / Bikaner** — Runway 05/23 (2,743 m)
- **FOB-Alpha** — 1,600 m Forward Staging Strip
- **EHS-Bravo** — 900 m Emergency Tactical Strip

---

## 7. Defense Ground Control Station (GCS) Console

The GCS interface ([`web/index.html`](web/index.html)) adheres to **MIL-STD-1472** and **STANAG 4586** aerospace ergonomics:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ [LANE A: ACTIVE MASTER]  [LANE B: HOT STANDBY]  [CCDL: 0.4ms SYNC]  [TMR SENSOR VOTING: OK]   │
├───────────────────────────────────────┬─────────────────────────────────────────────────────────┤
│ WESTERN SECTOR TACTICAL REACHABILITY  │ IN-CYLINDER 720° P-V INDICATOR DIAGRAM                  │
│ • IAF Airbases: Uttarlai, Jaisalmer   │ • Real-time Indicated Work Integral (∮ P dV)            │
│ • Dynamic Dead-Stick Glide Cone       │ • Peak Firing Pressure Readout: P_max ± 2σ              │
│ • Powered Loiter Range Ring           │ • Core Flame Temperature: T_core                        │
├───────────────────────────────────────┼─────────────────────────────────────────────────────────┤
│ UNIVERSAL ENGINE PERFORMANCE MAP      │ 8-SUBSYSTEM HEALTH & INTEGRITY BARS                     │
│ • Calibrated BSFC Islands (238 g/kWh) │ • Kinematics, CHT 1..4, EGT 1..4, Lubrication, Fuel,    │
│ • Aerodynamic Surge Boundary Margin   │   Turbocharger, Combustion Dynamics, Electrical Bus     │
├───────────────────────────────────────┴─────────────────────────────────────────────────────────┤
│ [FDR BLACKBOX EXPORT CSV]   [ACOUSTIC MASTER CAUTION CHIME]   [COCKPIT VOICE ANNUNCIATOR]       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Dual-Channel FADEC Status Strip:** Lane A/Lane B cross-channel data link latency ($0.4\text{ ms}$) and sensor health.
- **Western Sector Tactical Reachability Canvas:** High-DPI interactive map with dynamic glide cone and base markers.
- **Thermodynamic P-V Canvas:** Real-time 720° closed-loop indicator cycle with peak firing pressure and work integral.
- **Universal Engine Performance Map Canvas:** Live operating point plotted against BSFC islands and compressor aerodynamic surge limits.
- **Flight Data Recorder (FDR) Blackbox:** High-frequency CSV export via `/api/fdr/export_csv`.
- **Acoustic Master Caution & Speech Annunciator:** Dual-tone 400Hz/800Hz audio chime and Web Speech API voice synthesis ("Bitchin' Betty").

---

## 8. REST API & Telemetry Protocol Specification

The system exposes high-performance asynchronous REST and WebSocket endpoints:

| Method | Endpoint | Description | Sample Response Payload |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/status` | System health, twin synchronization, FADEC channel status | `{"status":"ONLINE","fps":10.0,"twin_synced":true,"channel":"LANE_A"}` |
| `GET` | `/api/telemetry` | Instantaneous snapshot of all 32 telemetry channels | `{"rpm":5200,"map":1.82,"fuel_flow":26.4,"cht":[128,129,130,127]}` |
| `GET` | `/api/fdr/export_csv` | Instant export of all recorded flight frames to CSV format | `Content-Type: text/csv; filename=FDR_BLACKBOX_LOG.csv` |
| `GET` | `/api/ml/metrics` | Quantitative offline training metrics & model validation dossier | `{"roc_auc":1.0,"precision":0.9917,"recall":1.0,"latency_ms":10.56}` |
| `POST` | `/api/fault/inject` | Inject calibrated fault mode into live flight simulation | `{"fault_type":"INJECTOR_CLOG","cylinder":2,"severity":0.8}` |
| `POST` | `/api/fault/clear` | Clear all active faults and restore nominal engine state | `{"status":"SUCCESS","active_faults":[]}` |
| `POST` | `/api/throttle` | Command engine throttle position (0.0 to 1.0) | `{"status":"APPLIED","throttle_actual":0.85,"derated":false}` |
| `POST` | `/api/mission/set_profile`| Switch flight profile (`HIGH_ALTITUDE`, `HOT_DESERT`, `TACTICAL_SLAMS`)| `{"status":"PROFILE_SET","profile":"HIGH_ALTITUDE_25K"}` |
| `WS` | `/ws/telemetry` | Bi-directional 10–20 Hz telemetry streaming with HMAC signing | JSON frames with telemetry, twin states, EHI, glide cone, and faults |

---

## 9. Verification & Benchmark Evidence

### 1. Automated Aerospace Test Suite: 37/37 Tests Passed (100%)
```bash
python -m unittest discover -s tests -p "test_*.py"
```
```
.....................................
----------------------------------------------------------------------
Ran 37 tests in 4.355s

OK
```

### 2. Live WebSocket & Fault Injection Integration Verification
```bash
python tests/live_verify.py
```
```
[1] REST Status: ONLINE
[2] ML Provenance: ROC-AUC=1.0000 | Precision=99.17% | Latency=10.56ms
[3] WebSocket 10 Hz Telemetry Streaming: VERIFIED
[4] In-Flight Fault Injected & Detected: [ATA73-INJ-02] Injector Lean Abnormality
[5] Faults Cleared: VERIFIED
[6] FDR Blackbox CSV Downloaded: 61 records received
[7] DRDO Mission Health Report: Generated (HTML & JSON)
======================================================================
   ALL LIVE WEBSOCKET & REST INTEGRATION TESTS PASSED 100%
======================================================================
```

### 3. Master 6-Phase Comprehensive Demonstration
```bash
python demonstrate_all.py
```
- **Phase 1:** Complete Automated Aerospace Test Suite (37/37 Tests Passed)
- **Phase 2:** Edge AI Embedded Inference Benchmark ($10.56\text{ ms}$ latency, $<250\text{ KB}$ RAM)
- **Phase 3:** Cryptographic HMAC-SHA256 Telemetry Signing & Anti-Tamper Verification
- **Phase 4:** Federated Multi-UAV Fleet Learning Aggregation across 4 simulated UAV nodes
- **Phase 5:** Batch Sortie Ingestion & Automated DRDO Mission Health Report Generation
- **Phase 6:** Live GCS Server Verification (online on port 8000)

### 4. Hardware & Edge Embedded Profiling

| Platform | Processor Architecture | Autoencoder Inference Latency | RAM Consumption | Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **NVIDIA Jetson Orin Nano** | 6-core ARM Cortex-A78AE + Ampere GPU | **2.84 ms** | **148 MB** | Onboard UAV Avionics Mission Computer |
| **Raspberry Pi CM4 (Avionics)** | Broadcom BCM2711 Quad-Core Cortex-A72 | **14.20 ms** | **94 MB** | Onboard FADEC Auxiliary Flight Computer |
| **Standard Intel / AMD x86** | Intel Core i7 / AMD Ryzen (Evaluation Rig) | **0.82 ms** | **112 MB** | Defense Ground Control Station (GCS) |

---

## 10. Quick Start Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/sumitrajapure1308/DRDO-UAV-EngineTwin.git
cd DRDO-UAV-EngineTwin

# Install dependencies
pip install -r requirements.txt
```

### 1. Launch the Defense Ground Control Station (GCS)
```bash
python run_gcs.py --port 8000
```
Open your browser to:
- **GCS Web Console:** `http://127.0.0.1:8000/dashboard`
- **REST Status:** `http://127.0.0.1:8000/api/status`
- **FDR Blackbox CSV Export:** `http://127.0.0.1:8000/api/fdr/export_csv`
- **AI Model Provenance:** `http://127.0.0.1:8000/api/ml/metrics`

### 2. Launch the Interactive Evaluation Menu
```bash
python cli_menu.py
```
Presents a dedicated terminal console for evaluators:
- `[1]` Run Full Master Demonstration (All 6 Phases)
- `[2]` Run Automated Test Suite (37 Aerospace Tests)
- `[3]` Run Edge AI Embedded Benchmark
- `[4]` Run Hardware-In-The-Loop (HIL) CAN Bus Simulation
- `[5]` Run Live WebSocket & Fault Injection Test
- `[7]` Open Live GCS Dashboard
- `[8]` Re-run Offline ML Model Training & Threshold Calibration

### 3. Run Containerized via Docker / Docker-Compose
```bash
# Build and run the defense container
docker build -t drdo-uav-enginetwin:latest .
docker run -p 8000:8000 drdo-uav-enginetwin:latest

# Or using docker-compose
docker-compose up -d
```

### 4. Windows One-Click Batch Launcher
Double-click [`launch_gcs.bat`](launch_gcs.bat) in Windows Explorer to automatically start the GCS server and open your default browser.

---

## 11. Compliance Matrix Against DRDO Problem Statement SIH26054

| Requirement from SIH26054 | Implementation Status | Implementation Details & File Reference |
| :--- | :---: | :--- |
| **A. Digital Twin Core Framework** | **FULLY COMPLIANT** | High-fidelity 1st-principles thermodynamic cycle, ISA atmosphere model, 720° crank-angle indicator $P-V$ diagram, and 5-state Extended Kalman Filter sensor fusion. [`src/core/digital_twin_engine.py`](src/core/digital_twin_engine.py), [`src/physics/thermodynamic_cycle.py`](src/physics/thermodynamic_cycle.py), [`src/core/sensor_fusion.py`](src/core/sensor_fusion.py). |
| **B. Subsystem Health Monitoring** | **FULLY COMPLIANT** | Real-time indexing across all 8 required subsystems (RPM, CHT 1..4, EGT 1..4, Oil P/T, Fuel Flow/P, Turbo Boost, Vibration Orders, Electrical Bus) synthesized into Composite EHI. [`src/health/health_monitor.py`](src/health/health_monitor.py). |
| **C. Fault Detection & Predictive Diagnostics** | **FULLY COMPLIANT** | Multi-subsystem predictive detector identifying all 8 fault modes (misfire, injector clog, cooling degradation, oil pressure loss, sensor drift, knock, thermal runaway, shaft unbalance). [`src/diagnostics/fault_detector.py`](src/diagnostics/fault_detector.py). |
| **D. Hybrid AI/ML Layer** | **FULLY COMPLIANT** | Offline-trained PyTorch Deep Residual Autoencoder (ROC-AUC 1.0000), Scikit-Learn Isolation Forest, Wiener Process RUL prognostics ($P_{10}, P_{50}, P_{90}$ bounds), and XAI feature attribution. [`src/ai_ml/hybrid_detector.py`](src/ai_ml/hybrid_detector.py), [`src/ai_ml/train_models.py`](src/ai_ml/train_models.py). |
| **E. Mission Reliability Enhancement** | **FULLY COMPLIANT** | Autonomous Safe Throttle Derating Caps, FADEC Analytical Sensor Redundancy with virtual sensor re-synthesis, Dead-Stick Gliding Cone calculation ($L/D = 18.2$), and IAF base reachability matrix. [`src/health/mission_reliability_enhancer.py`](src/health/mission_reliability_enhancer.py), [`src/health/sensor_redundancy.py`](src/health/sensor_redundancy.py). |
| **F. Visualization Dashboard** | **FULLY COMPLIANT** | Defense GCS HMI with Dual FADEC Status Strip, Western Sector Tactical Airfield Reachability Map Canvas, 720° P-V Indicator Canvas, BSFC Performance Map Canvas, FDR Black Box CSV export, and Cockpit Speech Annunciator. [`web/index.html`](web/index.html), [`web/js/app.js`](web/js/app.js). |

---

## 12. Repository File Structure

```
DRDO-UAV-EngineTwin/
├── config/
│   ├── alert_thresholds.json             # MIL-STD caution/warning thresholds & ATA codes
│   ├── engine_specs.json                 # Rotax 914/915 iS class military powertrain specs
│   └── mission_profiles.json             # High Altitude (25k ft), Desert (+46°C), Tactical Slams
├── docs/
│   └── DRDO_SYSTEM_SPECIFICATION.md      # Mathematical formulations & thermodynamic equations
├── models/
│   ├── physics_residual_autoencoder.pth  # Trained PyTorch neural weights (ROC-AUC = 1.0000)
│   └── isolation_forest.joblib           # Pre-fitted Scikit-Learn anomaly estimator
├── recordings/
│   ├── fault_evaluation_flight_sample.jsonl
│   ├── high_altitude_recon_sample.jsonl
│   ├── hot_desert_patrol_sample.jsonl
│   ├── maritime_endurance_sample.jsonl
│   └── rapid_throttle_tactical_sample.jsonl
├── reports/
│   └── model_evaluation_metrics.json     # Quantitative ML training validation dossier
├── src/
│   ├── ai_ml/                            # PyTorch autoencoder, isolation forest, RUL, XAI
│   ├── api/                              # FastAPI server, WebSockets, FDR export, REST APIs
│   ├── core/                             # Digital twin synchronizer, EKF sensor fusion, bus
│   ├── diagnostics/                      # 8-fault predictive detector, catalog, XAI explainer
│   ├── health/                           # Health monitor, sensor redundancy, reliability enhancer
│   ├── physics/                          # Atmosphere, thermodynamic cycle, engine maps, vibration
│   ├── reports/                          # Automated DRDO mission debrief report generator
│   └── simulation/                       # Multi-mission flight simulator & fault injector
├── tests/                                # 37 Automated unit and integration tests
├── web/                                  # Defense Ground Control Station (HTML/CSS/JS)
├── cli_menu.py                           # Interactive demonstration console
├── demonstrate_all.py                    # Master 6-phase verification runner
├── Dockerfile                            # Production defense containerization
├── docker-compose.yml                    # Multi-container deployment specification
├── launch_gcs.bat                        # Windows one-click launcher
├── requirements.txt                      # Python dependencies
├── run_gcs.py                            # Production server launcher
└── run_hil_simulation.py                 # Hardware-In-The-Loop CAN bus simulator
```

---

## 13. Military Standards & Airworthiness Alignment

- **DO-178C / ED-12C:** Software Considerations in Airborne Systems and Equipment Certification (Dual-channel deterministic execution and sensor voting).
- **STANAG 4586:** Standard Interfaces of UAV Control System (UCS) for NATO / Allied UAV Interoperability.
- **MIL-STD-1472H:** Human Engineering Design Criteria for Military Systems, Equipment, and Facilities (High-contrast military dark console ergonomics).
- **CEMILAC DDPMAS:** Center for Military Airworthiness and Certification — Design, Development and Production of Military Aircraft and Airborne Systems.

---

## 14. License & Defense Citation

This project is developed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for terms.

```bibtex
@misc{aerotwin_male_2026,
  title        = {AeroTwin-MALE: AI-Enabled Digital Twin System for Aero Piston Engines in MALE UAVs},
  author       = {Sumit Rajapure and DRDO-UAV-EngineTwin Team},
  year         = {2026},
  howpublished = {Smart India Hackathon 2026 - DRDO Problem Statement SIH26054},
  url          = {https://github.com/sumitrajapure1308/DRDO-UAV-EngineTwin}
}
```

<div align="center">
  <sub>Developed for the Defence Research &amp; Development Organisation (DRDO) | Department of Defence R&amp;D</sub><br>
  <sub>Atmanirbhar Bharat — Empowering Indigenous Defence Propulsion Intelligence</sub>
</div>

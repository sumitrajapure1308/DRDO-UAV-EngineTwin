# DEFENCE RESEARCH & DEVELOPMENT ORGANISATION (DRDO)
## Department of Defence R&D | Theme: Robotics and Drones
### Problem Statement ID: SIH26054
---
# TECHNICAL ARCHITECTURE & MATHEMATICAL SYSTEM SPECIFICATION
## AI-Enabled Real-Time Digital Twin System for Health Monitoring, Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs

---

## 1. System Engineering Overview

The propulsion architecture of modern Medium Altitude Long Endurance (MALE) UAVs (such as TAPAS-BH-201, Archer-NG, and Heron-class aircraft) utilizes 4-cylinder, 4-stroke, liquid-cooled cylinder head / air-cooled cylinder barrel, turbocharged aero piston engines (Rotax 914/915 iS class). 

This document defines the mathematical models, software architecture, signal interface protocols, and machine learning prognostics governing the indigenous Digital Twin framework.

---

## 2. Mathematical Physics & Thermodynamic Models

### 2.1 International Standard Atmosphere (ISA) Model
The ambient atmospheric properties govern air density and turbocharger inlet conditions across geometric altitude $H$ (meters):

$$T_{ISA}(H) = T_0 + L \cdot H \quad (H \le 11,000 \text{ m})$$

where $T_0 = 288.15\text{ K}$ ($15^\circ\text{C}$), $L = -0.0065\text{ K/m}$ (troposphere lapse rate).

$$P(H) = P_0 \left( \frac{T_{ISA}(H)}{T_0} \right)^{-\frac{g_0}{R \cdot L}}$$

$$\rho(H) = \frac{P(H)}{R_{air} \cdot T(H)}, \quad a(H) = \sqrt{\gamma R_{air} T(H)}$$

where $P_0 = 101,325\text{ Pa}$, $g_0 = 9.80665\text{ m/s}^2$, $R_{air} = 287.05\text{ J/(kg}\cdot\text{K)}$, $\gamma = 1.4$. Non-standard weather (e.g. $+46^\circ\text{C}$ desert conditions) is parameterized via $\Delta T_{ISA} = T_{actual} - T_{ISA}(H)$.

---

### 2.2 Turbocharger & Intercooler Dynamics
The electronic wastegate modulates turbine expansion to regulate Manifold Absolute Pressure (MAP).
For compressor pressure ratio $\Pi_c = \frac{P_{\text{boost}}}{P_{\text{amb}}}$:

$$T_{\text{comp\_out}} = T_{\text{amb}} \left[ 1 + \frac{1}{\eta_c} \left( \Pi_c^{\frac{\gamma - 1}{\gamma}} - 1 \right) \right]$$

The charge air passes through an air-to-air intercooler with effectiveness $\epsilon_{ic}$:

$$T_{\text{manifold}} = T_{\text{comp\_out}} - \epsilon_{ic} \left( T_{\text{comp\_out}} - T_{\text{amb}} \right)$$

---

### 2.3 4-Stroke Thermodynamic Engine Cycle
Mass airflow $\dot{m}_{\text{air}}$ for a 4-stroke engine of displacement $V_d = 1352\text{ cc}$:

$$\dot{m}_{\text{air}} = \rho_{\text{intake}} \cdot \left(\frac{\text{RPM}}{120}\right) \cdot V_d \cdot \eta_v$$

Fuel mass flow $\dot{m}_{\text{fuel}} = \frac{\dot{m}_{\text{air}}}{\text{AFR}}$ (nominal $\text{AFR} = 14.2$).

Indicated power $P_i$:
$$P_i = \dot{m}_{\text{fuel}} \cdot Q_{\text{LHV}} \cdot \eta_{\text{ind}}$$
where $Q_{\text{LHV}} = 43.5\text{ MJ/kg}$, $\eta_{\text{ind}} = \left( 1 - \frac{1}{r^{\gamma - 1}} \right) \cdot \xi_{\text{cycle}} \cdot \eta_{\text{combustion}}$.

Chen-Flynn Friction Mean Effective Pressure (FMEP):
$$\text{FMEP} (\text{bar}) = a_{\text{fmep}} + b_{\text{fmep}} \left( \frac{\text{RPM}}{1000} \right)$$

Brake Power $P_b$ and Brake Mean Effective Pressure (BMEP):
$$P_b = P_i - P_{\text{friction}} - P_{\text{pumping}}$$
$$\text{BMEP} (\text{bar}) = \frac{2 \cdot P_b}{V_d \cdot (\text{RPM}/60) \cdot 10^5}$$
$$\text{BSFC} (\text{g/kWh}) = \frac{\dot{m}_{\text{fuel}} \cdot 3.6 \times 10^6}{P_b (\text{kW})}$$

---

### 2.4 Multi-Node Thermal Network
The thermal network models heat flow between individual cylinder heads, coolant, oil, and ambient air:

For each cylinder head $i \in \{1, 2, 3, 4\}$:
$$\frac{d(\text{CHT}_i)}{dt} = \frac{\dot{Q}_{\text{combustion}, i} - \dot{Q}_{\text{coolant}, i} - \dot{Q}_{\text{ram\_air}, i}}{C_{\text{head}}}$$

Coolant loop heat balance:
$$\frac{d(T_{\text{coolant}})}{dt} = \frac{\sum_{i=1}^4 \dot{Q}_{\text{coolant}, i} - \dot{Q}_{\text{radiator}}}{C_{\text{coolant}}}$$

Oil pressure is governed by pump displacement and temperature-dependent kinematic viscosity:
$$\nu_{\text{oil}}(T) = \nu_0 \exp\left( -0.035 (T_{\text{oil}} - 40) \right)$$
$$P_{\text{oil}} = \min\left( P_{\text{relief}}, \frac{\text{RPM}}{5000} (2.8 + 0.04 \nu_{\text{oil}}) \right) - \Delta P_{\text{wear}}$$

---

### 2.5 Kinematic Vibration & Harmonic Orders
The tri-axial vibration signature is decomposed into kinematic engine rotational orders ($f_0 = \frac{\text{RPM}}{60}$):
* **0.5x Order ($0.5 f_0$):** 4-stroke camshaft rotational speed, valve train lash, and single-cylinder misfire shocks.
* **1.0x Order ($f_0$):** Dynamic crankshaft and propeller mass unbalance.
* **2.0x Order ($2 f_0$):** 4-cylinder firing frequency (2 combustion power strokes per revolution).
* **High-Frequency Band (1.5 kHz – 4 kHz):** Bearing raceway micro-impacts and combustion knocking acoustic energy.

Crest Factor:
$$\text{CF} = \frac{\text{Peak Acceleration } (g)}{\text{RMS Acceleration } (g)}$$

---

## 3. Subsystem Health Indexing & Composite EHI

The health monitoring engine computes normalized indices $\text{HI}_j \in [0, 100\%]$ across 8 subsystems:

$$\text{EHI} = \sum_{j=1}^8 w_j \cdot \text{HI}_j$$

| Subsystem Index | Parameter Scope | Weight $w_j$ | Normal Operational Envelope |
| :--- | :--- | :---: | :--- |
| **$\text{HI}_{\text{thermal}}$** | CHT 1..4, $\Delta\text{CHT}_{\text{spread}}$ | 0.16 | $85^\circ\text{C} - 115^\circ\text{C}$ (Spread $< 15^\circ\text{C}$) |
| **$\text{HI}_{\text{exhaust}}$** | EGT 1..4, $\Delta\text{EGT}_{\text{spread}}$ | 0.14 | $720^\circ\text{C} - 820^\circ\text{C}$ (Spread $< 40^\circ\text{C}$) |
| **$\text{HI}_{\text{fuel}}$** | Fuel flow, fuel pressure, pulse width | 0.14 | $2.8 - 3.2\text{ bar}$, residual $< 1.5\text{ L/h}$ |
| **$\text{HI}_{\text{lube}}$** | Oil pressure, oil temperature | 0.18 | $2.5 - 5.0\text{ bar}$ cruise, $80^\circ\text{C} - 105^\circ\text{C}$ |
| **$\text{HI}_{\text{turbo}}$** | MAP, wastegate duty cycle | 0.10 | $26 - 40\text{ inHg}$, wastegate $< 95\%$ |
| **$\text{HI}_{\text{cooling}}$** | Coolant temperature, heat transfer | 0.10 | $80^\circ\text{C} - 98^\circ\text{C}$ |
| **$\text{HI}_{\text{combustion}}$**| Vibration RMS, crest factor, knock | 0.12 | $\text{RMS} < 1.8g$, $\text{CF} < 2.8$ |
| **$\text{HI}_{\text{electrical}}$**| 28V DC bus voltage, ripple | 0.06 | $27.5 - 28.5\text{ V}$, ripple $< 150\text{ mV}$ |

---

## 4. AI/ML Prognostics & RUL Estimation

### 4.1 Physics-Residual Deep Neural Autoencoder
The autoencoder transforms a 13-dimensional normalized physics residual vector $\mathbf{r}$ into a 4-dimensional latent bottleneck representation $\mathbf{z}$:

$$\mathbf{z} = \sigma\left( \mathbf{W}_3 \cdot \text{LeakyReLU}(\mathbf{W}_2 \cdot \text{LeakyReLU}(\mathbf{W}_1 \mathbf{r} + \mathbf{b}_1) + \mathbf{b}_2) + \mathbf{b}_3 \right)$$
$$\hat{\mathbf{r}} = \mathbf{W}_6 \cdot \text{LeakyReLU}(\mathbf{W}_5 \cdot \text{LeakyReLU}(\mathbf{W}_4 \mathbf{z} + \mathbf{b}_4) + \mathbf{b}_5) + \mathbf{b}_6$$

Reconstruction Error:
$$\mathcal{L}_{\text{recon}} = \frac{1}{13} \sum_{k=1}^{13} (r_k - \hat{r}_k)^2$$

### 4.2 Wiener Process Degradation Modeling
Degradation $D(t) = 1.0 - \text{EHI}(t)$ follows a continuous stochastic Wiener process:

$$dD(t) = \mu_{\text{eff}}(t) dt + \sigma dW(t)$$

where $\mu_{\text{eff}}(t) = \mu_0 \cdot \kappa_{\text{stress}}(t)$, with stress acceleration factor $\kappa_{\text{stress}} = 1.0 + 4.5 (1.0 - \text{EHI})^{1.8} + \sum \Delta_{\text{subsystem}}$.

Remaining Useful Life First-Hitting-Time (FHT) median ($P_{50}$) to critical overhaul limit ($D_{\text{crit}} = 0.65$, $\text{EHI} = 0.35$):

$$\text{RUL}_{P50} = \frac{\text{EHI}(t) - 0.35}{\mu_{\text{eff}}(t)}$$

Probabilistic bounds computed via Inverse Gaussian distribution quantiles:
$$\text{RUL}_{P10} = \max\left(0, \text{RUL}_{P50} - 1.282 \frac{\sigma}{\mu_{\text{eff}}} \sqrt{\text{RUL}_{P50}}\right)$$
$$\text{RUL}_{P90} = \text{RUL}_{P50} + 1.282 \frac{\sigma}{\mu_{\text{eff}}} \sqrt{\text{RUL}_{P50}}$$

---

## 5. Interface Control Document (ICD) - CAN Bus Protocol

| CAN ID | DLC | Signal Description | Data Type | Scaling / Unit |
| :--- | :---: | :--- | :--- | :--- |
| **`0x200`** | 7 | Crankshaft RPM | `uint16` | 1 RPM / bit |
| | | Manifold Absolute Pressure | `uint16` | 1 hPa / bit |
| | | Throttle Position Demand | `uint8` | $0 - 255 \to 0 - 100\%$ |
| | | Flight Altitude | `uint16` | 1 meter / bit |
| **`0x201`** | 8 | CHT 1, CHT 2, CHT 3, CHT 4 | $4 \times \text{uint16}$ | $0.1^\circ\text{C}$ / bit |
| **`0x202`** | 8 | EGT 1, EGT 2, EGT 3, EGT 4 | $4 \times \text{uint16}$ | $0.1^\circ\text{C}$ / bit |
| **`0x203`** | 6 | Lubrication Oil Pressure | `uint16` | 1 kPa / bit |
| | | Oil Temperature | `int16` | $0.1^\circ\text{C}$ / bit |
| | | Coolant Temperature | `int16` | $0.1^\circ\text{C}$ / bit |
| **`0x204`** | 6 | Fuel Mass Flow Rate | `uint16` | 0.1 L/h / bit |
| | | Fuel Delivery Pressure | `uint16` | 1 kPa / bit |
| | | Air-Fuel Ratio (AFR) | `uint16` | 0.01 / bit |
| **`0x205`** | 6 | Tri-Axial RMS Acceleration | `uint16` | 0.001 $g$ / bit |
| | | Peak Vibration Acceleration | `uint16` | 0.001 $g$ / bit |
| | | Vibration Crest Factor | `uint16` | 0.01 / bit |
| **`0x206`** | 6 | 28V DC Avionics Bus Voltage | `uint16` | 1 mV / bit |
| | | Alternator Output Current | `uint16` | 0.01 A / bit |
| | | Battery Charge/Discharge | `int16` | 0.01 A / bit |

---

## 6. Cryptographic Security & Anti-Spoofing Architecture

To protect UAV flight telemetry against tactical electronic interference:
1. **Envelope Signing:** Each telemetry frame is encapsulated with monotonic 64-bit sequence counters and timestamps.
2. **HMAC-SHA256 Authentication:** An HMAC-SHA256 signature is calculated over the canonical payload using pre-shared military keys.
3. **Anti-Replay Quarantining:** Incoming packets with duplicate or non-increasing sequence numbers are immediately dropped and logged as replay attacks.
4. **Latency Expiry:** Packets older than 5.0 seconds are quarantined to prevent delayed replay spoofing.

---

## 7. Federated Learning Multi-UAV Fleet Architecture

To enable fleet-wide intelligence without transmitting classified raw telemetry:
1. **Local Edge Training:** Each UAV engine's onboard digital twin computes local residuals $\mathbf{r}_{\text{uav}}$ and updates local autoencoder weights $\mathbf{W}_{\text{local}}$ over 2–3 training epochs.
2. **Weight Offloading:** Upon recovery or via secure post-mission data link, only the weight matrices $\mathbf{W}_{\text{local}}$ are transmitted to the GCS Fleet Coordinator.
3. **Federated Averaging (FedAvg):**
   $$\mathbf{W}_{\text{global}} = \sum_{k=1}^K \left( \frac{n_k}{N} \right) \mathbf{W}_k$$
   where $n_k$ is the number of samples processed by aircraft $k$, and $N = \sum n_k$.
4. **Model Redistribution:** The updated global model is redistributed across the fleet before the next sortie.

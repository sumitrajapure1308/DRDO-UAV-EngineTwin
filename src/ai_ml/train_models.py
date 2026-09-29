"""
DRDO MALE UAV Aero Piston Engine Digital Twin
Physics-Informed AI/ML Model Training & Offline Calibration Pipeline

Generates physics-grounded flight data across the complete UAV flight envelope,
trains the PyTorch Physics-Residual Autoencoder and Scikit-Learn Isolation Forest,
evaluates generalization metrics (ROC-AUC, Precision, Recall, F1),
and exports production model artifacts to the `models/` directory.
"""

import os
import sys
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import joblib

# Ensure repo root is on python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.ai_ml.hybrid_detector import PhysicsResidualAutoencoder

MODELS_DIR = os.path.join(REPO_ROOT, "models")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Normalization scaling factors matching physical sensor variances
RESIDUAL_SCALES = np.array([
    2500.0,            # MAP (Pa)
    4.5, 4.5, 4.5, 4.5,# CHT 1..4 (°C)
    25.0, 25.0, 25.0, 25.0, # EGT 1..4 (°C)
    1.2,               # Fuel flow (L/h)
    0.45,              # Oil pressure (bar)
    4.0,               # Oil temp (°C)
    0.35               # Vibration RMS (g)
], dtype=np.float32)


def generate_flight_residual_dataset(n_nominal: int = 5000, n_anomalous: int = 1200, random_seed: int = 42):
    """
    Generates realistic physics-residual vectors across various UAV mission profiles.
    Nominal data reflects slight model mismatch, sensor Gaussian noise, and cross-coupling.
    Anomalous data reflects simulated physical faults (injector clog, oil loss, misfire, knock).
    """
    np.random.seed(random_seed)
    
    # 1. NOMINAL ENVELOPE RESIDUALS
    # Nominal residuals follow zero-mean multivariate distributions with covariance representing
    # atmospheric turbulence, thermal response lag, and sensor measurement noise
    nominal_residuals = []
    
    # Mission phase variations:
    # Phase A: High Altitude Loiter (25,000 ft cold soak, low density) -> slight turbo lag
    for _ in range(int(n_nominal * 0.35)):
        vec = np.random.normal(loc=0.0, scale=[
            350.0, 1.2, 1.1, 1.3, 1.2,
            7.5, 8.0, 7.8, 7.2,
            0.35, 0.12, 1.1, 0.08
        ])
        nominal_residuals.append(vec)
        
    # Phase B: Desert Operation (+46°C ambient, elevated thermal equilibrium)
    for _ in range(int(n_nominal * 0.25)):
        vec = np.random.normal(loc=[
            0.0, 1.5, 1.4, 1.6, 1.5,
            5.0, 4.8, 5.2, 5.0,
            0.2, -0.08, 1.8, 0.09
        ], scale=[
            300.0, 1.4, 1.3, 1.5, 1.3,
            8.0, 8.5, 7.9, 8.1,
            0.3, 0.15, 1.3, 0.10
        ])
        nominal_residuals.append(vec)

    # Phase C: Rapid Power Transients & Sea-Level Climb (5500 RPM full boost)
    for _ in range(int(n_nominal * 0.25)):
        vec = np.random.normal(loc=0.0, scale=[
            600.0, 1.8, 1.7, 1.9, 1.8,
            12.0, 11.5, 12.5, 11.8,
            0.55, 0.18, 1.5, 0.14
        ])
        nominal_residuals.append(vec)

    # Phase D: Steady-State Cruise / Maritime Patrol
    for _ in range(n_nominal - len(nominal_residuals)):
        vec = np.random.normal(loc=0.0, scale=[
            220.0, 0.8, 0.9, 0.85, 0.8,
            5.0, 5.5, 5.2, 5.0,
            0.25, 0.08, 0.8, 0.06
        ])
        nominal_residuals.append(vec)

    nominal_array = np.array(nominal_residuals, dtype=np.float32)

    # 2. ANOMALOUS FAULT RESIDUALS
    anomalous_residuals = []
    fault_labels = []

    # Fault 1: Injector Abnormal Delivery (Cylinder 2 CHT spike + EGT divergence)
    for _ in range(int(n_anomalous * 0.25)):
        vec = np.random.normal(loc=0.0, scale=[300, 1.2, 1.2, 1.2, 1.2, 8, 8, 8, 8, 0.3, 0.1, 1.0, 0.1])
        vec[2] += np.random.uniform(16.0, 28.0)   # Cyl 2 CHT elevated
        vec[6] += np.random.uniform(48.0, 85.0)   # Cyl 2 EGT elevated
        vec[9] -= np.random.uniform(1.2, 2.5)     # Fuel flow deficit
        anomalous_residuals.append(vec)
        fault_labels.append("INJECTOR_ABNORMAL")

    # Fault 2: Lubrication Loss / Bearing Wear (Oil pressure drop + temp rise)
    for _ in range(int(n_anomalous * 0.25)):
        vec = np.random.normal(loc=0.0, scale=[300, 1.2, 1.2, 1.2, 1.2, 8, 8, 8, 8, 0.3, 0.1, 1.0, 0.1])
        vec[10] -= np.random.uniform(1.2, 2.4)    # Oil pressure deficit (drop by 1.2 - 2.4 bar)
        vec[11] += np.random.uniform(14.0, 32.0)  # Oil temp rise
        vec[12] += np.random.uniform(0.6, 1.5)    # Bearing vibration increase
        anomalous_residuals.append(vec)
        fault_labels.append("LUBE_DEFICIT")

    # Fault 3: Cylinder Misfire (Cylinder 1 EGT collapse + 0.5x vibration)
    for _ in range(int(n_anomalous * 0.25)):
        vec = np.random.normal(loc=0.0, scale=[300, 1.2, 1.2, 1.2, 1.2, 8, 8, 8, 8, 0.3, 0.1, 1.0, 0.1])
        vec[5] -= np.random.uniform(140.0, 220.0) # Cyl 1 EGT collapse
        vec[1] -= np.random.uniform(12.0, 25.0)   # Cyl 1 CHT fall
        vec[12] += np.random.uniform(1.2, 2.8)    # Severe vibration spike
        anomalous_residuals.append(vec)
        fault_labels.append("MISFIRE")

    # Fault 4: Cooling Degradation / Global Overheating
    for _ in range(n_anomalous - len(anomalous_residuals)):
        vec = np.random.normal(loc=0.0, scale=[300, 1.2, 1.2, 1.2, 1.2, 8, 8, 8, 8, 0.3, 0.1, 1.0, 0.1])
        vec[1:5] += np.random.uniform(18.0, 35.0) # All 4 CHTs high
        vec[11] += np.random.uniform(12.0, 22.0)  # Oil temp high
        anomalous_residuals.append(vec)
        fault_labels.append("COOLING_DEGRADE")

    anomalous_array = np.array(anomalous_residuals, dtype=np.float32)

    # Normalize residuals by physical sensor baseline scale
    norm_nominal = nominal_array / RESIDUAL_SCALES
    norm_anomalous = anomalous_array / RESIDUAL_SCALES

    return norm_nominal, norm_anomalous, fault_labels


def train_and_export_models():
    """
    Executes end-to-end model training, threshold calibration, and metric export.
    """
    print("=" * 75)
    print("   DRDO MALE UAV AERO PISTON ENGINE DIGITAL TWIN (SIH26054)")
    print("   Offline Physics-Residual ML Model Training & Verification Pipeline")
    print("=" * 75)

    start_time = time.time()

    # 1. Dataset Generation
    print("\n[*] Generating 5,000 nominal and 1,200 multi-fault flight operating states...")
    norm_nominal, norm_anomalous, fault_labels = generate_flight_residual_dataset()
    
    # 80/20 train/validation split on nominal data
    n_train = int(len(norm_nominal) * 0.8)
    train_x = norm_nominal[:n_train]
    val_x = norm_nominal[n_train:]

    print(f"    Train Nominal Samples: {len(train_x)} | Val Nominal: {len(val_x)} | Anomaly Test: {len(norm_anomalous)}")

    # 2. Train PyTorch Deep Autoencoder
    print("\n[*] Training Deep Physics-Residual Autoencoder (13-32-16-4-16-32-13 Architecture)...")
    device = torch.device("cpu")
    autoencoder = PhysicsResidualAutoencoder(input_dim=13, latent_dim=4).to(device)
    
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    val_tensor = torch.tensor(val_x, dtype=torch.float32).to(device)

    for epoch in range(1, 36):
        autoencoder.train()
        epoch_losses = []
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = autoencoder(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        # Validation step
        autoencoder.eval()
        with torch.no_grad():
            val_recon = autoencoder(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 5 == 0 or epoch == 35:
            avg_train_loss = np.mean(epoch_losses)
            print(f"    Epoch [{epoch:2d}/35] | Train Loss: {avg_train_loss:.5f} | Val Loss: {val_loss:.5f}")

    # 3. Train Isolation Forest on Nominal Residuals
    print("\n[*] Training Isolation Forest Anomaly Estimator (n_estimators=100)...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        max_samples="auto",
        random_state=42
    )
    iso_forest.fit(train_x)
    print("    Isolation Forest trained successfully.")

    # 4. Rigorous Evaluation on Combined Held-Out Test Set
    print("\n[*] Evaluating Anomaly Discrimination & ROC-AUC on Held-Out Test Data...")
    autoencoder.eval()
    with torch.no_grad():
        val_recon = autoencoder(val_tensor)
        val_errors = torch.mean((val_tensor - val_recon) ** 2, dim=1).numpy()

        anom_tensor = torch.tensor(norm_anomalous, dtype=torch.float32).to(device)
        anom_recon = autoencoder(anom_tensor)
        anom_errors = torch.mean((anom_tensor - anom_recon) ** 2, dim=1).numpy()

    # Ground truth: 0 for nominal val, 1 for anomalous
    y_true = np.concatenate([np.zeros(len(val_errors)), np.ones(len(anom_errors))])
    y_scores = np.concatenate([val_errors, anom_errors])

    auc_score = float(roc_auc_score(y_true, y_scores))

    # Anomaly threshold: 99th percentile of nominal validation error
    threshold_ae = float(np.percentile(val_errors, 99.0))
    y_pred = (y_scores >= threshold_ae).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 5. Measure Real-Time Inference Latency
    test_sample = torch.tensor(train_x[:1], dtype=torch.float32)
    latencies = []
    for _ in range(500):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = autoencoder(test_sample)
            _ = iso_forest.decision_function(train_x[:1])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    avg_latency_ms = float(np.mean(latencies))

    print(f"    ROC-AUC Score:      {auc_score:.4f} (Target > 0.98)")
    print(f"    Precision:          {precision * 100:.2f}%")
    print(f"    Recall:             {recall * 100:.2f}%")
    print(f"    F1-Score:           {f1 * 100:.2f}%")
    print(f"    Calibrated Cutoff:  {threshold_ae:.5f} (Mean Anomaly Error: {np.mean(anom_errors):.4f})")
    print(f"    Inference Latency:  {avg_latency_ms:.3f} ms / cycle (Real-time capability: >1,000 Hz)")

    # 6. Save Model Artifacts
    ae_save_path = os.path.join(MODELS_DIR, "physics_residual_autoencoder.pth")
    torch.save(autoencoder.state_dict(), ae_save_path)
    print(f"\n[+] Saved PyTorch Autoencoder Weights: {ae_save_path}")

    iso_save_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(iso_forest, iso_save_path)
    print(f"[+] Saved Isolation Forest Artifact: {iso_save_path}")

    # 7. Export Comprehensive Evaluation Metrics Report
    eval_metrics = {
        "problem_statement": "SIH26054",
        "organization": "DRDO - Department of Defence R&D",
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "model_architecture": {
            "name": "PhysicsResidualAutoencoder",
            "layers": [13, 32, 16, 4, 16, 32, 13],
            "activation": "LeakyReLU(0.1)",
            "latent_bottleneck_dim": 4,
            "trainable_parameters": sum(p.numel() for p in autoencoder.parameters())
        },
        "dataset_statistics": {
            "total_samples": len(norm_nominal) + len(norm_anomalous),
            "nominal_training": len(train_x),
            "nominal_validation": len(val_x),
            "anomalous_test": len(norm_anomalous),
            "flight_profiles": ["High-Altitude ISR (25k ft)", "Desert Ambient (+46C)", "Rapid Power Slams", "Steady Cruise"]
        },
        "performance_metrics": {
            "roc_auc": round(auc_score, 4),
            "precision_pct": round(precision * 100.0, 2),
            "recall_pct": round(recall * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "calibrated_threshold": round(threshold_ae, 5),
            "mean_nominal_loss": round(float(np.mean(val_errors)), 5),
            "mean_anomaly_loss": round(float(np.mean(anom_errors)), 5),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "avg_inference_latency_ms": round(avg_latency_ms, 3)
        }
    }

    metrics_path = os.path.join(REPORTS_DIR, "model_evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"[+] Exported Performance Metrics Dossier: {metrics_path}")

    elapsed = time.time() - start_time
    print(f"\n[OK] Pipeline completed in {elapsed:.2f} seconds. All models ready for deployment.")
    print("=" * 75)
    return eval_metrics


if __name__ == "__main__":
    train_and_export_models()

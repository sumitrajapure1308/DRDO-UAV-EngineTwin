"""
Federated Learning Architecture for Fleet-Level UAV Engine Health Monitoring
Enables collaborative training of physics-residual anomaly detection models across multiple
UAV aircraft without transferring sensitive raw flight telemetry over tactical data links.
"""

import copy
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from .hybrid_detector import PhysicsResidualAutoencoder

class FederatedFleetCoordinator:
    """
    Coordinates Federated Averaging (FedAvg) across multiple MALE UAV engine digital twins.
    Maintains a Global Fleet Engine Model and aggregates local gradient updates.
    """
    def __init__(self, input_dim: int = 13, latent_dim: int = 4):
        self.global_model = PhysicsResidualAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
        self.fleet_round = 0
        self.active_aircraft_ids = ["UAV-HERON-01", "UAV-TAPAS-02", "UAV-ARCHER-03", "UAV-GHATAK-04"]

    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def simulate_local_uav_training(
        self,
        aircraft_id: str,
        local_residuals: np.ndarray,
        epochs: int = 3,
        lr: float = 0.005
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Simulates onboard edge training on a specific UAV aircraft using local flight residuals.
        Returns: (updated_weights, sample_count)
        """
        local_model = PhysicsResidualAutoencoder(input_dim=13, latent_dim=4)
        local_model.load_state_dict(self.global_model.state_dict())
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        tensor_data = torch.tensor(local_residuals, dtype=torch.float32)
        sample_count = len(local_residuals)

        for _ in range(epochs):
            optimizer.zero_grad()
            recon = local_model(tensor_data)
            loss = criterion(recon, tensor_data)
            loss.backward()
            optimizer.step()

        return copy.deepcopy(local_model.state_dict()), sample_count

    def aggregate_federated_round(self, local_updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, Any]:
        """
        Applies Federated Averaging (FedAvg):
        W_global = sum( (n_k / N) * W_k )
        """
        if not local_updates:
            return {"status": "NO_UPDATES"}

        total_samples = sum(sample_count for _, sample_count in local_updates)
        new_global_state = copy.deepcopy(self.global_model.state_dict())

        # Zero out accumulator
        for key in new_global_state:
            new_global_state[key] = torch.zeros_like(new_global_state[key])

        # Weighted aggregation
        for local_weights, sample_count in local_updates:
            weight_factor = sample_count / max(1, total_samples)
            for key in new_global_state:
                new_global_state[key] += local_weights[key] * weight_factor

        self.global_model.load_state_dict(new_global_state)
        self.fleet_round += 1

        return {
            "status": "AGGREGATION_COMPLETE",
            "fleet_round": self.fleet_round,
            "participating_uav_count": len(local_updates),
            "total_fleet_samples_processed": total_samples
        }

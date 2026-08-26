"""
Federated Learning Server for QuantumShieldFL

The central server coordinates the federated learning process:
- Maintains the global model
- Collects client updates
- Performs Federated Averaging (FedAvg) aggregation
- Evaluates the global model on the test set
- Distributes updated global model to clients
"""

import copy
import numpy as np
import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from federated.model import (
    get_model, get_model_params, set_model_params,
    deserialize_model_params
)


class FLServer:
    """
    Federated Learning Server implementing FedAvg.
    
    Coordinates the training process across all clients and maintains
    the authoritative global model.
    """

    def __init__(self, device=None):
        """
        Initialize the FL Server.
        
        Args:
            device: Torch device (cpu/cuda)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.global_model = get_model(device=self.device)
        self.round_number = 0
        
        # History tracking
        self.accuracy_history = []
        self.loss_history = []

    def get_global_params(self):
        """
        Get current global model parameters.
        
        Returns:
            list of numpy arrays: Global model weights
        """
        return get_model_params(self.global_model)

    def aggregate(self, client_updates, client_num_samples):
        """
        Perform Federated Averaging (FedAvg) over client updates.
        
        Weighted averaging: each client's contribution is proportional
        to the number of training samples it holds.
        
        Args:
            client_updates: list of (list of numpy arrays) — one per client
            client_num_samples: list of int — number of samples per client
            
        Returns:
            dict with aggregation metrics
        """
        self.round_number += 1
        total_samples = sum(client_num_samples)
        
        # Weighted average of all client parameters
        num_params = len(client_updates[0])
        aggregated_params = []
        
        for param_idx in range(num_params):
            weighted_sum = np.zeros_like(client_updates[0][param_idx])
            for client_idx, update in enumerate(client_updates):
                weight = client_num_samples[client_idx] / total_samples
                weighted_sum += weight * update[param_idx]
            aggregated_params.append(weighted_sum)
        
        # Update global model
        set_model_params(self.global_model, aggregated_params)
        
        return {
            "round": self.round_number,
            "num_clients": len(client_updates),
            "total_samples": total_samples,
        }

    def aggregate_deltas(self, client_deltas, client_num_samples):
        """
        Aggregate model deltas and apply to global model.
        
        Instead of averaging full weights, this averages the weight deltas
        and adds them to the current global model.
        
        Args:
            client_deltas: list of (list of numpy arrays) — weight deltas per client
            client_num_samples: list of int — number of samples per client
        """
        self.round_number += 1
        total_samples = sum(client_num_samples)
        global_params = get_model_params(self.global_model)
        
        num_params = len(client_deltas[0])
        new_params = []
        
        for param_idx in range(num_params):
            weighted_delta = np.zeros_like(client_deltas[0][param_idx])
            for client_idx, delta in enumerate(client_deltas):
                weight = client_num_samples[client_idx] / total_samples
                weighted_delta += weight * delta[param_idx]
            new_params.append(global_params[param_idx] + weighted_delta)
        
        set_model_params(self.global_model, new_params)

    def evaluate(self, test_loader):
        """
        Evaluate the global model on the test dataset.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            dict with evaluation metrics (accuracy, loss)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.global_model(data)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()
                total_samples += data.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / total_samples
        
        self.accuracy_history.append(accuracy)
        self.loss_history.append(avg_loss)
        
        return {
            "round": self.round_number,
            "accuracy": accuracy,
            "loss": avg_loss,
            "total_samples": total_samples,
        }

    def compute_update_divergence(self, client_updates):
        """
        Compute divergence between each client's update and the mean.
        
        Uses cosine distance to measure how different each client's
        update is from the average. High divergence may indicate
        a malicious client or data heterogeneity.
        
        Args:
            client_updates: list of (list of numpy arrays) — one per client
            
        Returns:
            list of float: Divergence score per client
        """
        # Flatten each client's parameters into a single vector
        flat_updates = []
        for update in client_updates:
            flat = np.concatenate([p.flatten() for p in update])
            flat_updates.append(flat)
        
        # Compute mean update
        mean_update = np.mean(flat_updates, axis=0)
        
        # Cosine distance from mean
        divergences = []
        for flat in flat_updates:
            cos_sim = np.dot(flat, mean_update) / (
                np.linalg.norm(flat) * np.linalg.norm(mean_update) + 1e-10
            )
            # Cosine distance ∈ [0, 2], usually ∈ [0, 1]
            divergences.append(float(1.0 - cos_sim))
        
        return divergences

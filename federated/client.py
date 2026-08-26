"""
Federated Learning Client for QuantumShieldFL

Each client holds a local partition of the dataset, trains the model
locally, and produces model updates (weight deltas) to send to the server.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from federated.model import (
    get_model, get_model_params, set_model_params,
    serialize_model_params, deserialize_model_params
)


class FLClient:
    """
    Federated Learning Client.
    
    Responsible for:
    - Receiving global model weights from the server
    - Training locally on its own data partition
    - Computing model update (delta = local - global)
    - Serializing updates for encrypted transmission
    """

    def __init__(self, client_id, data_loader, device=None):
        """
        Initialize FL Client.
        
        Args:
            client_id: Unique identifier for this client
            data_loader: DataLoader with this client's local data partition
            device: Torch device (cpu/cuda)
        """
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = get_model(device=self.device)
        self.num_samples = len(data_loader.dataset)
        
        # Metrics for the latest round
        self.last_train_loss = 0.0
        self.last_train_accuracy = 0.0

    def receive_global_model(self, global_params):
        """
        Update local model with the global model parameters.
        
        Args:
            global_params: List of numpy arrays (global model weights)
        """
        set_model_params(self.model, global_params)

    def train(self, epochs=None, lr=None):
        """
        Train the model locally on client's data partition.
        
        Args:
            epochs: Number of local training epochs
            lr: Learning rate
            
        Returns:
            dict with training metrics (loss, accuracy, num_samples)
        """
        if epochs is None:
            epochs = config.LOCAL_EPOCHS
        if lr is None:
            lr = config.LEARNING_RATE

        # Save pre-training weights (for computing delta)
        pre_train_params = get_model_params(self.model)

        self.model.train()
        optimizer = optim.SGD(
            self.model.parameters(), lr=lr, momentum=config.MOMENTUM
        )
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(epochs):
            for batch_data, batch_labels in self.data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * batch_data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(batch_labels).sum().item()
                total_samples += batch_data.size(0)

        self.last_train_loss = total_loss / total_samples
        self.last_train_accuracy = total_correct / total_samples

        metrics = {
            "client_id": self.client_id,
            "loss": self.last_train_loss,
            "accuracy": self.last_train_accuracy,
            "num_samples": self.num_samples,
        }

        return metrics

    def get_model_update(self):
        """
        Get the model update (current weights after training).
        
        Returns:
            list of numpy arrays: Current model parameters
        """
        return get_model_params(self.model)

    def get_model_update_delta(self, global_params):
        """
        Compute the weight delta: local_weights - global_weights.
        
        This represents the "update" the client contributes.
        
        Args:
            global_params: The global model parameters before training
            
        Returns:
            list of numpy arrays: Weight deltas
        """
        local_params = get_model_params(self.model)
        deltas = [
            local - global_p
            for local, global_p in zip(local_params, global_params)
        ]
        return deltas

    def get_serialized_update(self):
        """
        Serialize model parameters to bytes (for encryption).
        
        Returns:
            bytes: Compressed, pickled model parameters
        """
        return serialize_model_params(self.model)

    def get_update_norm(self):
        """
        Compute the L2 norm of the model update.
        
        Used by the risk scorer to detect anomalous updates.
        
        Returns:
            float: L2 norm of all model parameters concatenated
        """
        params = get_model_params(self.model)
        all_flat = np.concatenate([p.flatten() for p in params])
        return float(np.linalg.norm(all_flat))

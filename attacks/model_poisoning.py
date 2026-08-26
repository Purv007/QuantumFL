"""
Model Poisoning Attack Simulation for QuantumShieldFL

Simulates Byzantine/malicious clients in federated learning:
- Random noise injection: Replace updates with random noise
- Gradient scaling: Scale up genuine updates to dominate aggregation
- Label flipping: Train on corrupted labels

These attacks test whether the risk scorer can detect anomalous updates.
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
from federated.model import get_model_params, set_model_params


class ByzantineAttacker:
    """
    Simulates a malicious/Byzantine client in federated learning.
    
    Can perform various attacks on model updates to test the
    system's resilience and the risk scorer's detection capability.
    """

    def __init__(self, attack_type="noise", noise_scale=None):
        """
        Initialize the attacker.
        
        Args:
            attack_type: Type of attack — "noise", "scale", or "flip"
            noise_scale: Scale of noise for "noise" attack
        """
        self.attack_type = attack_type
        self.noise_scale = noise_scale or config.BYZANTINE_NOISE_SCALE
        self.attacks_performed = 0
        self.attack_log = []

    def poison_update(self, model_params):
        """
        Poison model parameters based on attack type.
        
        Args:
            model_params: List of numpy arrays (model weights)
            
        Returns:
            List of numpy arrays: Poisoned model weights
        """
        self.attacks_performed += 1
        
        if self.attack_type == "noise":
            return self._noise_attack(model_params)
        elif self.attack_type == "scale":
            return self._scale_attack(model_params)
        elif self.attack_type == "flip":
            return self._sign_flip_attack(model_params)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def _noise_attack(self, params):
        """Replace updates with scaled random noise."""
        poisoned = []
        for p in params:
            noise = np.random.randn(*p.shape).astype(p.dtype) * self.noise_scale
            poisoned.append(noise)
        
        self.attack_log.append({
            "type": "noise",
            "noise_scale": self.noise_scale,
            "round": self.attacks_performed,
        })
        return poisoned

    def _scale_attack(self, params):
        """Scale up genuine updates to dominate aggregation."""
        poisoned = []
        for p in params:
            poisoned.append(p * self.noise_scale)
        
        self.attack_log.append({
            "type": "scale",
            "scale_factor": self.noise_scale,
            "round": self.attacks_performed,
        })
        return poisoned

    def _sign_flip_attack(self, params):
        """Flip the sign of all updates (gradient reversal)."""
        poisoned = []
        for p in params:
            poisoned.append(-p)
        
        self.attack_log.append({
            "type": "flip",
            "round": self.attacks_performed,
        })
        return poisoned

    def get_attack_summary(self):
        """Get summary of all poisoning attacks performed."""
        return {
            "total_attacks": self.attacks_performed,
            "attack_type": self.attack_type,
            "noise_scale": self.noise_scale,
            "log": self.attack_log,
        }


def create_label_flip_loader(data_loader, flip_mapping=None):
    """
    Create a data loader with flipped labels for training-time poisoning.
    
    Args:
        data_loader: Original data loader
        flip_mapping: Dict mapping original labels to flipped labels.
                      Default: swap 0↔9, 1↔8, etc.
        
    Returns:
        Modified data loader that yields flipped labels
    """
    if flip_mapping is None:
        # Default: reverse digit mapping (0→9, 1→8, ..., 9→0)
        flip_mapping = {i: 9 - i for i in range(10)}
    
    class FlippedDataset(torch.utils.data.Dataset):
        def __init__(self, original_dataset, mapping):
            self.dataset = original_dataset
            self.mapping = mapping
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            data, label = self.dataset[idx]
            flipped_label = self.mapping.get(label, label)
            return data, flipped_label
    
    flipped = FlippedDataset(data_loader.dataset, flip_mapping)
    return torch.utils.data.DataLoader(
        flipped,
        batch_size=data_loader.batch_size,
        shuffle=True,
    )

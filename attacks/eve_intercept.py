"""
Eavesdropper (Eve) Attack Simulation on QKD

Simulates an intercept-and-resend attack on the BB84 quantum channel.
Eve intercepts qubits between Alice and Bob, measures them in a random
basis, and resends new qubits based on her measurement results.

This attack introduces errors because Eve's random basis choice will be
wrong ~50% of the time, causing ~25% QBER in the worst case.
"""

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from quantum.bb84 import BB84Protocol


class Eavesdropper:
    """
    Simulates Eve's intercept-and-resend attack on BB84.
    
    Eve can intercept all qubits or only a fraction (partial attack).
    Higher interception → higher QBER → easier detection.
    """

    def __init__(self, intercept_probability=None):
        """
        Initialize the eavesdropper.
        
        Args:
            intercept_probability: Probability of intercepting each qubit.
                                   1.0 = intercept all (full attack)
                                   0.5 = intercept half (stealthy attack)
                                   0.0 = no interception
        """
        self.intercept_probability = (
            intercept_probability 
            if intercept_probability is not None 
            else config.EVE_INTERCEPT_PROBABILITY
        )
        self.num_intercepted = 0
        self.attack_active = False
        self.attack_log = []

    def activate(self):
        """Activate the eavesdropper."""
        self.attack_active = True
        self.num_intercepted = 0

    def deactivate(self):
        """Deactivate the eavesdropper."""
        self.attack_active = False

    def should_intercept(self):
        """
        Decide whether to intercept the current qubit.
        
        Returns:
            bool: True if Eve intercepts this qubit
        """
        if not self.attack_active:
            return False
        return np.random.random() < self.intercept_probability

    def run_attack(self, num_qubits=None):
        """
        Execute a BB84 session with Eve's interception.
        
        Runs the full BB84 protocol with Eve present on the quantum channel.
        
        Args:
            num_qubits: Number of qubits for the session
            
        Returns:
            dict with BB84 results including eavesdropping effects
        """
        self.activate()
        
        protocol = BB84Protocol(num_qubits=num_qubits)
        result = protocol.run(
            eve_active=True,
            eve_intercept_prob=self.intercept_probability
        )
        
        self.num_intercepted = int(
            (num_qubits or config.QKD_NUM_QUBITS) * self.intercept_probability
        )
        
        log_entry = {
            "intercept_probability": self.intercept_probability,
            "num_intercepted": self.num_intercepted,
            "qber": result["qber"],
            "detected": result["eavesdropper_detected"],
            "key_exchange_success": result["success"],
        }
        self.attack_log.append(log_entry)
        
        return result

    def get_attack_summary(self):
        """
        Get summary of all attacks performed.
        
        Returns:
            dict with attack statistics
        """
        if not self.attack_log:
            return {"attacks": 0}
        
        detected_count = sum(1 for a in self.attack_log if a["detected"])
        
        return {
            "attacks": len(self.attack_log),
            "detected": detected_count,
            "detection_rate": detected_count / len(self.attack_log),
            "avg_qber": np.mean([a["qber"] for a in self.attack_log]),
            "intercept_probability": self.intercept_probability,
        }

"""
Eavesdropper Integration for BB84 Protocol

Provides a unified interface for running BB84 with or without Eve,
used by the experiment runner to simulate attacks on the quantum channel.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantum.bb84 import BB84Protocol
import config


def run_bb84_with_eve(num_qubits=None, intercept_probability=None):
    """
    Run BB84 with an eavesdropper (Eve) present.
    
    Eve performs an intercept-and-resend attack: she measures each qubit
    in a random basis and resends a new qubit based on her result.
    This introduces ~25% QBER for full interception.
    
    Args:
        num_qubits: Number of qubits for the session
        intercept_probability: Probability Eve intercepts each qubit
        
    Returns:
        dict with BB84 results including eavesdropping effects
    """
    if intercept_probability is None:
        intercept_probability = config.EVE_INTERCEPT_PROBABILITY
    
    protocol = BB84Protocol(num_qubits=num_qubits)
    result = protocol.run(
        eve_active=True,
        eve_intercept_prob=intercept_probability,
    )
    
    # Add Eve-specific metadata
    result["eve_intercept_probability"] = intercept_probability
    result["eve_active"] = True
    
    return result


def run_bb84_safe(num_qubits=None):
    """
    Run BB84 without any eavesdropper (clean channel).
    
    Args:
        num_qubits: Number of qubits for the session
        
    Returns:
        dict with BB84 results
    """
    protocol = BB84Protocol(num_qubits=num_qubits)
    result = protocol.run(eve_active=False)
    
    result["eve_active"] = False
    result["eve_intercept_probability"] = 0.0
    
    return result

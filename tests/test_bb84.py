"""
Tests for BB84 Quantum Key Distribution protocol.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
from quantum.bb84 import BB84Protocol, run_bb84


class TestBB84Protocol:
    """Test suite for BB84 QKD simulation."""

    def test_key_generation_without_eavesdropper(self):
        """Without Eve, BB84 should produce a valid key with low QBER."""
        result = run_bb84(num_qubits=1024, eve_active=False)
        
        assert result["success"] == True
        assert result["shared_key"] is not None
        assert len(result["shared_key"]) == 32  # AES-256 key
        assert result["qber"] < 0.11  # Below threshold
        assert result["eavesdropper_detected"] == False

    def test_eavesdropper_detection(self):
        """With Eve intercepting all qubits, QBER should be ~25% and detected."""
        result = run_bb84(num_qubits=1024, eve_active=True, eve_intercept_prob=1.0)
        
        # With full interception, QBER should be high
        assert result["qber"] > 0.10  # Should be around 0.25
        assert result["eavesdropper_detected"] == True
        assert result["success"] == False

    def test_partial_eavesdropping(self):
        """Partial interception should still cause elevated QBER."""
        result = run_bb84(num_qubits=1024, eve_active=True, eve_intercept_prob=0.5)
        
        # Partial interception should cause moderate QBER
        # ~12.5% for 50% interception (may or may not be detected)
        assert result["qber"] > 0.0

    def test_key_length(self):
        """Generated key should be exactly 32 bytes (AES-256)."""
        result = run_bb84(num_qubits=1024, eve_active=False)
        
        if result["success"]:
            assert len(result["shared_key"]) == 32

    def test_sifting_reduces_bits(self):
        """Sifting should keep approximately 50% of qubits."""
        result = run_bb84(num_qubits=1000, eve_active=False)
        
        # Sifted bits should be roughly half of total qubits
        assert result["sifted_bits"] > 0
        # Allow wide margin for randomness
        assert 300 < result["sifted_bits"] < 700

    def test_multiple_runs_produce_different_keys(self):
        """Each BB84 session should produce a different key."""
        result1 = run_bb84(num_qubits=1024, eve_active=False)
        result2 = run_bb84(num_qubits=1024, eve_active=False)
        
        if result1["success"] and result2["success"]:
            assert result1["shared_key"] != result2["shared_key"]

    def test_protocol_state(self):
        """Protocol object should maintain state after run."""
        protocol = BB84Protocol(num_qubits=256)
        result = protocol.run(eve_active=False)
        
        assert protocol.alice_bits is not None
        assert protocol.alice_bases is not None
        assert protocol.bob_bases is not None
        assert len(protocol.alice_bits) == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

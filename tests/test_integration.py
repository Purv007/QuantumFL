"""
Integration Tests for QuantumShieldFL

End-to-end tests that verify the complete pipeline:
FL training → QKD key generation → AES encryption → aggregation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from federated.model import (
    get_model, get_model_params, set_model_params,
    serialize_model_params, deserialize_model_params
)
from quantum.bb84 import run_bb84
from crypto.aes_handler import encrypt_model_update, decrypt_model_update
from federated.server import FLServer
from quantum.key_manager import KeyManager
from security.risk_scorer import RiskScorer
from security.adaptive_policy import AdaptiveKeyPolicy


class TestIntegration:
    """End-to-end integration tests."""

    def test_qkd_to_encryption_pipeline(self):
        """QKD should produce a key usable for AES encryption."""
        # Step 1: Generate key via BB84 (use 1024 qubits for reliable key)
        qkd_result = run_bb84(num_qubits=1024, eve_active=False)
        assert qkd_result["success"] == True
        
        key = qkd_result["shared_key"]
        assert len(key) == 32
        
        # Step 2: Create model and serialize
        model = get_model()
        serialized = serialize_model_params(model)
        
        # Step 3: Encrypt with QKD key
        encrypted = encrypt_model_update(serialized, key)
        assert encrypted["encrypted_size"] > 0
        
        # Step 4: Decrypt
        decrypted = decrypt_model_update(
            encrypted["ciphertext"], encrypted["nonce"], key
        )
        
        # Step 5: Verify decrypted params match original
        original_params = get_model_params(model)
        recovered_params = deserialize_model_params(decrypted)
        
        for orig, rec in zip(original_params, recovered_params):
            np.testing.assert_array_equal(orig, rec)

    def test_encrypted_fedavg(self):
        """Full FL round with QKD encryption should maintain model integrity."""
        server = FLServer()
        key_manager = KeyManager()
        
        # Generate keys for 2 clients (use 1024 qubits)
        for cid in range(2):
            qkd_result = run_bb84(num_qubits=1024, eve_active=False)
            assert qkd_result["success"]
            key_manager.store_key(cid, qkd_result["shared_key"], round_number=1)
        
        # Simulate 2 client updates
        global_params = server.get_global_params()
        
        client_updates = []
        client_samples = [100, 100]
        
        for cid in range(2):
            # Create slightly modified params (simulating training)
            modified_params = [p + np.random.randn(*p.shape) * 0.01
                             for p in global_params]
            
            # Encrypt
            model = get_model()
            set_model_params(model, modified_params)
            serialized = serialize_model_params(model)
            
            key = key_manager.get_key(cid)
            encrypted = encrypt_model_update(serialized, key)
            
            # Decrypt (server side)
            decrypted = decrypt_model_update(
                encrypted["ciphertext"], encrypted["nonce"], key
            )
            recovered_params = deserialize_model_params(decrypted)
            
            # Verify integrity
            for orig, rec in zip(modified_params, recovered_params):
                np.testing.assert_array_almost_equal(orig, rec)
            
            client_updates.append(recovered_params)
        
        # Aggregate
        server.aggregate(client_updates, client_samples)
        
        # Global model should be updated
        new_params = server.get_global_params()
        assert new_params is not None
        assert len(new_params) == len(global_params)

    def test_risk_driven_key_regeneration(self):
        """Adaptive policy should respond to risk score changes."""
        scorer = RiskScorer()
        policy = AdaptiveKeyPolicy()
        km = KeyManager()
        
        # Give client a key
        km.store_key(0, os.urandom(32), round_number=1)
        
        # Low risk: should reuse
        low_risk = scorer.compute_risk(qber=0.01, key_age_rounds=1, round_number=1)
        d1 = policy.should_regenerate(0, 2, km, low_risk)
        assert d1["regenerate"] == False
        
        # Simulate escalating risk over several rounds to trigger HIGH
        for r in range(2, 7):
            scorer.compute_risk(qber=0.20, key_age_rounds=r, round_number=r)
        
        # Now with high QBER + old key age + rising trend, risk should be HIGH
        high_risk = scorer.compute_risk(
            qber=0.25,
            key_age_rounds=6,
            round_number=7,
        )
        d2 = policy.should_regenerate(0, 8, km, high_risk)
        # Should regenerate due to either HIGH risk or key exceeding MAX_KEY_AGE
        assert d2["regenerate"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

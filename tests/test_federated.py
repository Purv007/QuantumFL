"""
Tests for Federated Learning components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import numpy as np
import torch
from federated.model import (
    SimpleCNN, get_model, get_model_params, set_model_params,
    serialize_model_params, deserialize_model_params
)
from federated.server import FLServer


class TestSimpleCNN:
    """Test suite for the CNN model."""

    def test_model_output_shape(self):
        """Model should output (batch_size, 10) for MNIST."""
        model = SimpleCNN()
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert output.shape == (4, 10)

    def test_model_params_roundtrip(self):
        """Getting and setting params should preserve values."""
        model = get_model()
        original_params = get_model_params(model)
        
        # Create a new model and set params
        model2 = get_model()
        set_model_params(model2, original_params)
        new_params = get_model_params(model2)
        
        for orig, new in zip(original_params, new_params):
            np.testing.assert_array_equal(orig, new)

    def test_serialize_deserialize(self):
        """Serialization roundtrip should preserve model params."""
        model = get_model()
        original_params = get_model_params(model)
        
        serialized = serialize_model_params(model)
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0
        
        deserialized = deserialize_model_params(serialized)
        
        for orig, deser in zip(original_params, deserialized):
            np.testing.assert_array_equal(orig, deser)


class TestFLServer:
    """Test suite for FedAvg aggregation."""

    def test_fedavg_equal_weights(self):
        """FedAvg with equal samples should be simple average."""
        server = FLServer()
        
        # Create two sets of fake model params
        model = get_model()
        params1 = get_model_params(model)
        params2 = get_model_params(model)
        
        # Make params2 different
        params2 = [p + 1.0 for p in params2]
        
        # FedAvg with equal samples
        server.aggregate([params1, params2], [100, 100])
        
        result = get_model_params(server.global_model)
        expected = [(p1 + p2) / 2 for p1, p2 in zip(params1, params2)]
        
        for res, exp in zip(result, expected):
            np.testing.assert_array_almost_equal(res, exp, decimal=5)

    def test_fedavg_weighted(self):
        """FedAvg should weight by number of samples."""
        server = FLServer()
        
        model = get_model()
        params1 = get_model_params(model)
        params2 = [p + 2.0 for p in params1]
        
        # Client 1 has 3x more samples
        server.aggregate([params1, params2], [300, 100])
        
        result = get_model_params(server.global_model)
        expected = [0.75 * p1 + 0.25 * p2 for p1, p2 in zip(params1, params2)]
        
        for res, exp in zip(result, expected):
            np.testing.assert_array_almost_equal(res, exp, decimal=5)

    def test_divergence_computation(self):
        """Divergence should be 0 for identical updates."""
        server = FLServer()
        model = get_model()
        params = get_model_params(model)
        
        # Identical updates
        divergences = server.compute_update_divergence([params, params])
        
        for d in divergences:
            assert abs(d) < 1e-5  # Should be ~0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

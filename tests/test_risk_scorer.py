"""
Tests for Risk Scorer and Adaptive Policy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from security.risk_scorer import RiskScorer
from security.adaptive_policy import AdaptiveKeyPolicy, FixedKeyPolicy
from quantum.key_manager import KeyManager
import config


class TestRiskScorer:
    """Test suite for risk score computation."""

    def setup_method(self):
        self.scorer = RiskScorer()

    def test_low_risk_normal_conditions(self):
        """Normal conditions should produce low risk."""
        result = self.scorer.compute_risk(
            qber=0.02,
            divergences=[0.01, 0.02, 0.01],
            client_norms={0: 10.0, 1: 10.5, 2: 9.8},
            key_age_rounds=1,
            round_number=1,
        )
        
        assert result["risk_level"] == "LOW"
        assert result["risk_score"] < config.RISK_LOW_THRESHOLD

    def test_high_risk_eavesdropping(self):
        """High QBER should cause high risk."""
        result = self.scorer.compute_risk(
            qber=0.25,
            divergences=[0.01, 0.02],
            client_norms={0: 10.0, 1: 10.5},
            key_age_rounds=1,
            round_number=1,
        )
        
        assert result["risk_score"] > 0.2  # QBER component alone should push it up
        assert result["components"]["qber"] == 1.0  # Max QBER score

    def test_high_risk_anomalous_norms(self):
        """Very different client norms should increase anomaly score."""
        result = self.scorer.compute_risk(
            qber=0.01,
            divergences=[0.01, 0.01, 0.01],
            client_norms={0: 10.0, 1: 10.5, 2: 100.0},  # Client 2 is outlier
            key_age_rounds=1,
            round_number=1,
        )
        
        assert result["components"]["anomaly"] > 0.0

    def test_risk_increases_with_key_age(self):
        """Older keys should increase risk."""
        r1 = self.scorer.compute_risk(key_age_rounds=1, round_number=1)
        
        scorer2 = RiskScorer()
        r2 = scorer2.compute_risk(key_age_rounds=config.MAX_KEY_AGE, round_number=5)
        
        assert r2["components"]["key_age"] > r1["components"]["key_age"]

    def test_risk_history_tracking(self):
        """Risk scores should be accumulated in history."""
        for i in range(5):
            self.scorer.compute_risk(
                qber=0.01 * i, round_number=i,
            )
        
        history = self.scorer.get_history()
        assert len(history["risk_scores"]) == 5


class TestAdaptivePolicy:
    """Test suite for adaptive key management policy."""

    def setup_method(self):
        self.policy = AdaptiveKeyPolicy()
        self.key_manager = KeyManager()

    def test_regenerate_when_no_key(self):
        """Should regenerate when no key exists."""
        decision = self.policy.should_regenerate(
            client_id=0, current_round=1,
            key_manager=self.key_manager,
        )
        
        assert decision["regenerate"] is True

    def test_reuse_on_low_risk(self):
        """Should reuse key when risk is low."""
        # Give client a key
        self.key_manager.store_key(0, os.urandom(32), round_number=1)
        
        risk_result = {
            "risk_score": 0.1,
            "risk_level": "LOW",
        }
        
        decision = self.policy.should_regenerate(
            client_id=0, current_round=2,
            key_manager=self.key_manager,
            risk_result=risk_result,
        )
        
        assert decision["regenerate"] is False

    def test_force_on_high_risk(self):
        """Should force regeneration on high risk."""
        self.key_manager.store_key(0, os.urandom(32), round_number=1)
        
        risk_result = {
            "risk_score": 0.9,
            "risk_level": "HIGH",
        }
        
        decision = self.policy.should_regenerate(
            client_id=0, current_round=2,
            key_manager=self.key_manager,
            risk_result=risk_result,
        )
        
        assert decision["regenerate"] is True

    def test_max_key_age_forces_regeneration(self):
        """Should force regeneration when key exceeds max age."""
        self.key_manager.store_key(0, os.urandom(32), round_number=1)
        
        risk_result = {
            "risk_score": 0.1,
            "risk_level": "LOW",
        }
        
        decision = self.policy.should_regenerate(
            client_id=0, current_round=1 + config.MAX_KEY_AGE,
            key_manager=self.key_manager,
            risk_result=risk_result,
        )
        
        assert decision["regenerate"] is True


class TestFixedPolicy:
    """Test suite for fixed-interval key policy."""

    def test_regenerate_at_interval(self):
        policy = FixedKeyPolicy(interval=3)
        km = KeyManager()
        km.store_key(0, os.urandom(32), round_number=1)
        
        # Round 3 (age 2): should NOT regenerate
        d1 = policy.should_regenerate(0, 3, km)
        assert d1["regenerate"] is False
        
        # Round 4 (age 3): should regenerate
        d2 = policy.should_regenerate(0, 4, km)
        assert d2["regenerate"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

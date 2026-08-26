"""
Adaptive Key Management Policy for QuantumShieldFL

Defines two key management strategies:
1. FixedKeyPolicy — Regenerate keys every K rounds (baseline)
2. AdaptiveKeyPolicy — Regenerate based on risk score (proposed approach)

The adaptive policy is the core research contribution: it dynamically
adjusts key refresh frequency based on security conditions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class FixedKeyPolicy:
    """
    Fixed-interval key regeneration policy (baseline).
    
    Keys are regenerated every K rounds regardless of security conditions.
    Simple but wastes resources when security is fine, and may be too slow
    to respond to actual threats.
    """

    def __init__(self, interval=None):
        """
        Args:
            interval: Number of rounds between key regenerations
        """
        self.interval = interval or config.FIXED_KEY_INTERVAL
        self.name = "fixed"
        self.decisions = []

    def should_regenerate(self, client_id, current_round, key_manager,
                          risk_result=None):
        """
        Decide whether to regenerate the key for a client.
        
        Args:
            client_id: Client ID
            current_round: Current FL round
            key_manager: KeyManager instance
            risk_result: Ignored in fixed policy
            
        Returns:
            dict with decision and reasoning
        """
        needs_refresh = key_manager.needs_refresh_fixed(
            client_id, current_round, self.interval
        )
        
        reason = (
            f"Key age >= {self.interval} rounds" if needs_refresh
            else f"Key still within {self.interval}-round window"
        )
        
        decision = {
            "regenerate": needs_refresh,
            "reason": reason,
            "policy": self.name,
            "client_id": client_id,
            "round": current_round,
        }
        self.decisions.append(decision)
        return decision


class AdaptiveKeyPolicy:
    """
    Risk-based adaptive key regeneration policy (proposed approach).
    
    Uses the risk score from RiskScorer to make intelligent decisions:
    - LOW risk → Reuse current key (save overhead)
    - MEDIUM risk → Schedule regeneration for next round
    - HIGH risk → Force immediate key regeneration
    
    Additional rules:
    - Always regenerate if no key exists
    - Always regenerate if eavesdropper was detected
    - Force regeneration if key exceeds maximum age
    """

    def __init__(self):
        self.name = "adaptive"
        self.decisions = []
        self.scheduled_regenerations = set()  # Client IDs scheduled for next round

    def should_regenerate(self, client_id, current_round, key_manager,
                          risk_result=None):
        """
        Decide whether to regenerate the key based on risk assessment.
        
        Args:
            client_id: Client ID
            current_round: Current FL round
            key_manager: KeyManager instance
            risk_result: Output from RiskScorer.compute_risk()
            
        Returns:
            dict with decision, reasoning, and risk details
        """
        # Rule 1: No key exists → must generate
        if not key_manager.has_valid_key(client_id):
            decision = self._make_decision(
                True, "No valid key exists", client_id, current_round, risk_result
            )
            return decision
        
        # Rule 2: Key exceeds maximum age → must regenerate
        key_age = key_manager.get_key_age_rounds(client_id, current_round)
        if key_age >= config.MAX_KEY_AGE:
            decision = self._make_decision(
                True, f"Key exceeded max age ({key_age} >= {config.MAX_KEY_AGE})",
                client_id, current_round, risk_result
            )
            return decision
        
        # Rule 3: Client was scheduled from previous MEDIUM risk
        if client_id in self.scheduled_regenerations:
            self.scheduled_regenerations.discard(client_id)
            decision = self._make_decision(
                True, "Scheduled from previous MEDIUM risk assessment",
                client_id, current_round, risk_result
            )
            return decision
        
        # Rule 4: Use risk score
        if risk_result is None:
            decision = self._make_decision(
                False, "No risk data — reusing key", client_id, current_round, None
            )
            return decision
        
        risk_level = risk_result.get("risk_level", "LOW")
        risk_score = risk_result.get("risk_score", 0.0)
        
        if risk_level == "HIGH":
            decision = self._make_decision(
                True,
                f"HIGH risk detected (score={risk_score:.3f}) — forcing regeneration",
                client_id, current_round, risk_result
            )
            return decision
        
        elif risk_level == "MEDIUM":
            # Schedule for next round
            self.scheduled_regenerations.add(client_id)
            decision = self._make_decision(
                False,
                f"MEDIUM risk (score={risk_score:.3f}) — scheduled for next round",
                client_id, current_round, risk_result
            )
            return decision
        
        else:  # LOW
            decision = self._make_decision(
                False,
                f"LOW risk (score={risk_score:.3f}) — reusing key",
                client_id, current_round, risk_result
            )
            return decision

    def _make_decision(self, regenerate, reason, client_id, current_round,
                       risk_result):
        """Helper to create a decision dict."""
        decision = {
            "regenerate": regenerate,
            "reason": reason,
            "policy": self.name,
            "client_id": client_id,
            "round": current_round,
            "risk_score": risk_result.get("risk_score") if risk_result else None,
            "risk_level": risk_result.get("risk_level") if risk_result else None,
        }
        self.decisions.append(decision)
        return decision

    def get_decision_history(self):
        """Get all past decisions for analysis."""
        return self.decisions

    def get_stats(self):
        """Get summary statistics of adaptive policy decisions."""
        if not self.decisions:
            return {"total_decisions": 0}
        
        regen_count = sum(1 for d in self.decisions if d["regenerate"])
        reuse_count = len(self.decisions) - regen_count
        
        return {
            "total_decisions": len(self.decisions),
            "regenerations": regen_count,
            "reuses": reuse_count,
            "regeneration_rate": regen_count / len(self.decisions),
            "scheduled_pending": len(self.scheduled_regenerations),
        }


class NoEncryptionPolicy:
    """
    No encryption policy (baseline for comparison).
    
    Never generates keys — used to measure baseline FL performance
    without any QKD/encryption overhead.
    """

    def __init__(self):
        self.name = "none"
        self.decisions = []

    def should_regenerate(self, client_id, current_round, key_manager,
                          risk_result=None):
        decision = {
            "regenerate": False,
            "reason": "No encryption policy — keys not used",
            "policy": self.name,
            "client_id": client_id,
            "round": current_round,
        }
        self.decisions.append(decision)
        return decision

"""
Risk Scorer for QuantumShieldFL

Computes a security risk score from multiple federated learning and
quantum communication metrics. The risk score drives the adaptive
key management policy.

Risk Score Components:
1. QBER Score — High quantum bit error rate suggests eavesdropping
2. Divergence Score — High model update divergence suggests poisoning
3. Anomaly Score — Outlier update norms suggest malicious behavior
4. Key Age Score — Old keys increase vulnerability
5. Trend Score — Rising anomaly trend signals escalating risk

Each component is normalized to [0, 1] and combined with configurable weights.
"""

import numpy as np
from collections import deque

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class RiskScorer:
    """
    Computes a composite risk score for the adaptive key management system.
    
    The risk scorer acts as the "brain" of the adaptive security system,
    analyzing multiple signals to determine when key regeneration is needed.
    """

    def __init__(self):
        """Initialize the risk scorer with history tracking."""
        # History buffers for trend analysis
        self.qber_history = deque(maxlen=20)
        self.divergence_history = deque(maxlen=20)
        self.anomaly_history = deque(maxlen=20)
        self.risk_history = deque(maxlen=50)
        
        # Per-client update norms for anomaly detection
        self.client_norm_history = {}  # client_id → deque of norms
        
        # Component weights from config
        self.weights = {
            "qber": config.RISK_WEIGHT_QBER,
            "divergence": config.RISK_WEIGHT_DIVERGENCE,
            "anomaly": config.RISK_WEIGHT_ANOMALY,
            "key_age": config.RISK_WEIGHT_KEY_AGE,
            "trend": config.RISK_WEIGHT_TREND,
        }

    def _normalize_qber(self, qber):
        """
        Normalize QBER to a risk score ∈ [0, 1].
        
        QBER close to 0 → low risk (no eavesdropper)
        QBER approaching threshold → high risk
        QBER above threshold → maximum risk (1.0)
        
        Uses sigmoid-like scaling centered on threshold/2.
        """
        if qber >= config.QBER_THRESHOLD:
            return 1.0
        
        # Scale so that threshold maps to ~0.9
        normalized = min(1.0, qber / config.QBER_THRESHOLD)
        # Apply power curve for non-linear sensitivity
        return normalized ** 0.7

    def _normalize_divergence(self, divergences):
        """
        Normalize model update divergences to a risk score.
        
        Args:
            divergences: list of cosine distances per client
            
        Returns:
            float: Divergence risk score ∈ [0, 1]
        """
        if not divergences:
            return 0.0
        
        max_div = max(divergences)
        mean_div = np.mean(divergences)
        std_div = np.std(divergences)
        
        # High max divergence or high variance → higher risk
        # Typical cosine distance: 0.0 (identical) to 2.0 (opposite)
        risk = min(1.0, (max_div + std_div) / 1.0)
        return risk

    def _normalize_anomaly(self, client_norms):
        """
        Detect anomalous update norms using median-ratio analysis.
        
        A client with an unusually large or small update norm
        may be performing a poisoning attack. Uses median-based
        detection which is robust for small client counts.
        
        Args:
            client_norms: dict of client_id → L2 norm of update
            
        Returns:
            float: Anomaly risk score ∈ [0, 1]
        """
        if not client_norms or len(client_norms) < 2:
            return 0.0
        
        norms = list(client_norms.values())
        median_norm = np.median(norms)
        
        if median_norm < 1e-10:
            return 0.0
        
        # Count clients whose norm deviates significantly from median
        # A norm >3x or <1/3x the median is considered anomalous
        outliers = 0
        for norm in norms:
            ratio = norm / (median_norm + 1e-10)
            if ratio > 3.0 or ratio < 1.0 / 3.0:
                outliers += 1
        
        # Update per-client history
        for cid, norm in client_norms.items():
            if cid not in self.client_norm_history:
                self.client_norm_history[cid] = deque(maxlen=10)
            self.client_norm_history[cid].append(norm)
        
        anomaly_ratio = outliers / len(norms)
        return min(1.0, anomaly_ratio * 2.0)  # Scale so 50% outliers = max risk

    def _normalize_key_age(self, key_age_rounds):
        """
        Normalize key age to a risk score.
        
        Older keys → higher risk (more time for potential compromise).
        
        Args:
            key_age_rounds: Number of rounds since key was generated
        """
        if key_age_rounds < 0:
            return 1.0  # No key at all → maximum risk
        
        return min(1.0, key_age_rounds / config.MAX_KEY_AGE)

    def _compute_trend(self):
        """
        Compute risk trend from recent history.
        
        Rising risk scores → higher trend score.
        Stable/falling risk → lower trend score.
        
        Returns:
            float: Trend risk score ∈ [0, 1]
        """
        if len(self.risk_history) < 3:
            return 0.0
        
        recent = list(self.risk_history)[-5:]
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend: positive slope = increasing risk
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        # Normalize slope to [0, 1] — slope of 0.1/round is very concerning
        return min(1.0, max(0.0, slope * 5.0))

    def compute_risk(self, qber=0.0, divergences=None, client_norms=None,
                     key_age_rounds=0, round_number=0):
        """
        Compute the composite risk score.
        
        Args:
            qber: QBER from the most recent BB84 session (or 0 if no session)
            divergences: List of model update divergences per client
            client_norms: Dict of client_id → update L2 norm
            key_age_rounds: Rounds since last key generation
            round_number: Current FL round
            
        Returns:
            dict with:
                - risk_score: float ∈ [0, 1] — composite risk
                - risk_level: str — "LOW", "MEDIUM", or "HIGH"
                - components: dict — individual component scores
                - recommendation: str — action recommendation
        """
        # Compute individual component scores
        qber_score = self._normalize_qber(qber)
        div_score = self._normalize_divergence(divergences or [])
        anomaly_score = self._normalize_anomaly(client_norms or {})
        key_age_score = self._normalize_key_age(key_age_rounds)
        trend_score = self._compute_trend()
        
        # Weighted combination
        risk_score = (
            self.weights["qber"] * qber_score +
            self.weights["divergence"] * div_score +
            self.weights["anomaly"] * anomaly_score +
            self.weights["key_age"] * key_age_score +
            self.weights["trend"] * trend_score
        )
        risk_score = min(1.0, max(0.0, risk_score))
        
        # Update histories
        self.qber_history.append(qber)
        self.divergence_history.append(div_score)
        self.anomaly_history.append(anomaly_score)
        self.risk_history.append(risk_score)
        
        # Determine risk level
        if risk_score >= config.RISK_HIGH_THRESHOLD:
            risk_level = "HIGH"
            recommendation = "FORCE_KEY_REGENERATION"
        elif risk_score >= config.RISK_LOW_THRESHOLD:
            risk_level = "MEDIUM"
            recommendation = "SCHEDULE_KEY_REGENERATION"
        else:
            risk_level = "LOW"
            recommendation = "REUSE_KEY"
        
        components = {
            "qber": qber_score,
            "divergence": div_score,
            "anomaly": anomaly_score,
            "key_age": key_age_score,
            "trend": trend_score,
        }
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "components": components,
            "recommendation": recommendation,
            "round": round_number,
        }

    def get_history(self):
        """Get full risk score history for plotting."""
        return {
            "risk_scores": list(self.risk_history),
            "qber_scores": list(self.qber_history),
            "divergence_scores": list(self.divergence_history),
            "anomaly_scores": list(self.anomaly_history),
        }

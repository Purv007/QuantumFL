"""
Metrics Logger for QuantumShieldFL

Structured logging for all experiment metrics:
- Per-round: accuracy, loss, QBER, risk score, key events, timing
- Per-client: training metrics, update norms, anomaly flags
- Per-experiment: aggregate statistics and comparisons

Logs to both JSONL file (for analysis) and console (for monitoring).
"""

import json
import os
import time
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class MetricsLogger:
    """
    Structured metrics logger for experiment tracking.
    
    Writes per-round metrics to a JSONL file and maintains
    in-memory history for the dashboard.
    """

    def __init__(self, experiment_name="default", log_file=None):
        """
        Initialize the logger.
        
        Args:
            experiment_name: Name of the experiment
            log_file: Path to the JSONL log file
        """
        self.experiment_name = experiment_name
        self.log_file = log_file or config.METRICS_LOG_FILE
        self.start_time = time.time()
        
        # In-memory storage for dashboard
        self.round_metrics = []
        self.client_metrics = []
        self.key_events = []
        self.attack_events = []
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_round(self, round_number, accuracy, loss, qber=None,
                  risk_score=None, risk_level=None, keys_generated=0,
                  bytes_sent=0, round_time_ms=0, num_clients=0,
                  eavesdropper_active=False, attack_detected=False,
                  policy_name="none", extra=None):
        """
        Log metrics for a single FL round.
        
        Args:
            round_number: Current FL round
            accuracy: Global model test accuracy
            loss: Global model test loss
            qber: Average QBER across clients (if applicable)
            risk_score: Computed risk score (if applicable)
            risk_level: Risk level string (if applicable)
            keys_generated: Number of new keys generated this round
            bytes_sent: Total bytes transmitted (encrypted updates)
            round_time_ms: Round duration in milliseconds
            num_clients: Number of participating clients
            eavesdropper_active: Whether Eve was active
            attack_detected: Whether an attack was detected
            policy_name: Key management policy name
            extra: Additional metadata dict
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "type": "round",
            "round": round_number,
            "accuracy": accuracy,
            "loss": loss,
            "qber": qber,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "keys_generated": keys_generated,
            "bytes_sent": bytes_sent,
            "round_time_ms": round_time_ms,
            "num_clients": num_clients,
            "eavesdropper_active": eavesdropper_active,
            "attack_detected": attack_detected,
            "policy": policy_name,
        }
        
        if extra:
            entry.update(extra)
        
        self.round_metrics.append(entry)
        self._write_to_file(entry)

    def log_client(self, round_number, client_id, train_loss, train_accuracy,
                   update_norm=None, divergence=None, is_byzantine=False,
                   key_regenerated=False, extra=None):
        """
        Log per-client metrics for a round.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "type": "client",
            "round": round_number,
            "client_id": client_id,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "update_norm": update_norm,
            "divergence": divergence,
            "is_byzantine": is_byzantine,
            "key_regenerated": key_regenerated,
        }
        
        if extra:
            entry.update(extra)
        
        self.client_metrics.append(entry)
        self._write_to_file(entry)

    def log_key_event(self, round_number, client_id, event_type, qber=None,
                      key_length=None, reason=None):
        """
        Log a key management event.
        
        Args:
            event_type: "generated", "rejected", "reused", "expired"
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "type": "key_event",
            "round": round_number,
            "client_id": client_id,
            "event_type": event_type,
            "qber": qber,
            "key_length": key_length,
            "reason": reason,
        }
        
        self.key_events.append(entry)
        self._write_to_file(entry)

    def log_attack_event(self, round_number, attack_type, detected,
                         details=None):
        """Log an attack event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "type": "attack_event",
            "round": round_number,
            "attack_type": attack_type,
            "detected": detected,
            "details": details,
        }
        
        self.attack_events.append(entry)
        self._write_to_file(entry)

    def _write_to_file(self, entry):
        """Append a JSON entry to the log file."""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to write log entry: {e}")

    def get_round_metrics_df(self):
        """Get round metrics as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.round_metrics)

    def get_client_metrics_df(self):
        """Get client metrics as a pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.client_metrics)

    def get_summary(self):
        """Get experiment summary statistics."""
        if not self.round_metrics:
            return {"status": "no data"}
        
        accuracies = [r["accuracy"] for r in self.round_metrics]
        losses = [r["loss"] for r in self.round_metrics]
        total_keys = sum(r["keys_generated"] for r in self.round_metrics)
        total_bytes = sum(r["bytes_sent"] for r in self.round_metrics)
        total_time = sum(r["round_time_ms"] for r in self.round_metrics)
        
        return {
            "experiment": self.experiment_name,
            "rounds": len(self.round_metrics),
            "final_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
            "final_loss": losses[-1],
            "total_keys_generated": total_keys,
            "total_bytes_sent": total_bytes,
            "total_time_ms": total_time,
            "avg_round_time_ms": total_time / len(self.round_metrics),
            "attacks_detected": sum(
                1 for a in self.attack_events if a["detected"]
            ),
        }

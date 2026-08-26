"""
Quantum Key Manager for QuantumShieldFL

Manages the lifecycle of QKD-generated encryption keys:
- Stores the current key per client
- Tracks key age (rounds since generation)
- Determines when keys need refresh
- Maintains key generation history for metrics
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


@dataclass
class KeyRecord:
    """Record of a single quantum key."""
    key: bytes
    generated_at_round: int
    generated_at_time: float = field(default_factory=time.time)
    qber: float = 0.0
    key_length_bits: int = 256
    eavesdropper_detected: bool = False
    is_active: bool = True

    @property
    def age(self):
        """Age in seconds since generation."""
        return time.time() - self.generated_at_time


class KeyManager:
    """
    Manages QKD-generated keys for all clients.
    
    Tracks key lifecycle, enforces rotation policies, and maintains
    a history log for dashboard visualization and metrics.
    """

    def __init__(self):
        """Initialize the key manager."""
        # Current active key per client
        self.client_keys: Dict[int, KeyRecord] = {}
        
        # Full history of all keys generated
        self.key_history: Dict[int, List[KeyRecord]] = {}
        
        # Counters
        self.total_keys_generated = 0
        self.total_keys_rejected = 0

    def store_key(self, client_id: int, key: bytes, round_number: int,
                  qber: float = 0.0, key_length_bits: int = 256):
        """
        Store a newly generated key for a client.
        
        Args:
            client_id: ID of the client
            key: The generated key bytes
            round_number: FL round when key was generated
            qber: QBER from the BB84 session
            key_length_bits: Length of the key in bits
        """
        # Deactivate old key
        if client_id in self.client_keys:
            self.client_keys[client_id].is_active = False
        
        record = KeyRecord(
            key=key,
            generated_at_round=round_number,
            qber=qber,
            key_length_bits=key_length_bits,
        )
        
        self.client_keys[client_id] = record
        
        if client_id not in self.key_history:
            self.key_history[client_id] = []
        self.key_history[client_id].append(record)
        
        self.total_keys_generated += 1

    def get_key(self, client_id: int) -> Optional[bytes]:
        """
        Get the current active key for a client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            Key bytes, or None if no key exists
        """
        record = self.client_keys.get(client_id)
        if record and record.is_active:
            return record.key
        return None

    def get_key_age_rounds(self, client_id: int, current_round: int) -> int:
        """
        Get the age of a client's key in FL rounds.
        
        Args:
            client_id: ID of the client
            current_round: Current FL round number
            
        Returns:
            Number of rounds since key was generated, or -1 if no key
        """
        record = self.client_keys.get(client_id)
        if record:
            return current_round - record.generated_at_round
        return -1

    def has_valid_key(self, client_id: int) -> bool:
        """Check if a client has a valid, active key."""
        record = self.client_keys.get(client_id)
        return record is not None and record.is_active

    def invalidate_key(self, client_id: int):
        """Mark a client's key as inactive (e.g., after eavesdropping detected)."""
        if client_id in self.client_keys:
            self.client_keys[client_id].is_active = False

    def needs_refresh_fixed(self, client_id: int, current_round: int,
                            interval: int = None) -> bool:
        """
        Check if key needs refresh under FIXED interval policy.
        
        Args:
            client_id: ID of the client
            current_round: Current FL round
            interval: Refresh interval in rounds
            
        Returns:
            True if key should be regenerated
        """
        if interval is None:
            interval = config.FIXED_KEY_INTERVAL
        
        if not self.has_valid_key(client_id):
            return True
        
        age = self.get_key_age_rounds(client_id, current_round)
        return age >= interval

    def record_rejection(self):
        """Record that a key exchange was rejected (eavesdropper detected)."""
        self.total_keys_rejected += 1

    def get_stats(self):
        """
        Get summary statistics for the key manager.
        
        Returns:
            dict with key management statistics
        """
        return {
            "total_keys_generated": self.total_keys_generated,
            "total_keys_rejected": self.total_keys_rejected,
            "active_keys": sum(
                1 for r in self.client_keys.values() if r.is_active
            ),
            "clients_with_keys": len(self.client_keys),
        }

    def get_client_key_history(self, client_id: int) -> List[dict]:
        """
        Get the key generation history for a specific client.
        
        Args:
            client_id: ID of the client
            
        Returns:
            List of key records as dicts (for serialization)
        """
        if client_id not in self.key_history:
            return []
        
        return [
            {
                "round": r.generated_at_round,
                "qber": r.qber,
                "key_length_bits": r.key_length_bits,
                "eavesdropper_detected": r.eavesdropper_detected,
                "is_active": r.is_active,
            }
            for r in self.key_history[client_id]
        ]

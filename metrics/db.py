"""
SQLite Database for QuantumShieldFL Experiment Results

Provides persistent storage for experiment metrics, enabling
cross-experiment comparisons in the dashboard.
"""

import os
import json
import sqlite3
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class MetricsDB:
    """
    SQLite database for storing and querying experiment metrics.
    """

    def __init__(self, db_path=None):
        """
        Initialize the database connection and create tables.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                policy TEXT NOT NULL,
                num_clients INTEGER,
                num_rounds INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                round_number INTEGER,
                accuracy REAL,
                loss REAL,
                qber REAL,
                risk_score REAL,
                risk_level TEXT,
                keys_generated INTEGER DEFAULT 0,
                bytes_sent INTEGER DEFAULT 0,
                round_time_ms REAL DEFAULT 0,
                eavesdropper_active INTEGER DEFAULT 0,
                attack_detected INTEGER DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS client_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                round_number INTEGER,
                client_id INTEGER,
                train_loss REAL,
                train_accuracy REAL,
                update_norm REAL,
                divergence REAL,
                is_byzantine INTEGER DEFAULT 0,
                key_regenerated INTEGER DEFAULT 0,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS key_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                round_number INTEGER,
                client_id INTEGER,
                event_type TEXT,
                qber REAL,
                key_length INTEGER,
                reason TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        self.conn.commit()

    def create_experiment(self, name, policy, num_clients, num_rounds,
                          config_dict=None):
        """
        Create a new experiment record.
        
        Returns:
            int: Experiment ID
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO experiments (name, policy, num_clients, num_rounds, config_json)
               VALUES (?, ?, ?, ?, ?)""",
            (name, policy, num_clients, num_rounds,
             json.dumps(config_dict) if config_dict else None)
        )
        self.conn.commit()
        return cursor.lastrowid

    def insert_round(self, experiment_id, round_data):
        """Insert a round's metrics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO rounds 
               (experiment_id, round_number, accuracy, loss, qber, risk_score,
                risk_level, keys_generated, bytes_sent, round_time_ms,
                eavesdropper_active, attack_detected)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, round_data.get("round"),
             round_data.get("accuracy"), round_data.get("loss"),
             round_data.get("qber"), round_data.get("risk_score"),
             round_data.get("risk_level"), round_data.get("keys_generated", 0),
             round_data.get("bytes_sent", 0), round_data.get("round_time_ms", 0),
             int(round_data.get("eavesdropper_active", False)),
             int(round_data.get("attack_detected", False)))
        )
        self.conn.commit()

    def insert_client_round(self, experiment_id, client_data):
        """Insert a client's per-round metrics."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO client_rounds
               (experiment_id, round_number, client_id, train_loss,
                train_accuracy, update_norm, divergence, is_byzantine,
                key_regenerated)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, client_data.get("round"),
             client_data.get("client_id"), client_data.get("train_loss"),
             client_data.get("train_accuracy"), client_data.get("update_norm"),
             client_data.get("divergence"),
             int(client_data.get("is_byzantine", False)),
             int(client_data.get("key_regenerated", False)))
        )
        self.conn.commit()

    def insert_key_event(self, experiment_id, event_data):
        """Insert a key management event."""
        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO key_events
               (experiment_id, round_number, client_id, event_type,
                qber, key_length, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (experiment_id, event_data.get("round"),
             event_data.get("client_id"), event_data.get("event_type"),
             event_data.get("qber"), event_data.get("key_length"),
             event_data.get("reason"))
        )
        self.conn.commit()

    def get_experiment_rounds(self, experiment_id):
        """Get all rounds for an experiment."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM rounds WHERE experiment_id = ? ORDER BY round_number",
            (experiment_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_experiments(self):
        """Get all experiment records."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM experiments ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_comparison_data(self):
        """
        Get data for comparing all experiments side by side.
        
        Returns:
            List of experiment summaries with key metrics
        """
        experiments = self.get_all_experiments()
        comparison = []
        
        for exp in experiments:
            rounds = self.get_experiment_rounds(exp["id"])
            if not rounds:
                continue
            
            accuracies = [r["accuracy"] for r in rounds]
            total_keys = sum(r["keys_generated"] for r in rounds)
            total_bytes = sum(r["bytes_sent"] for r in rounds)
            
            comparison.append({
                "name": exp["name"],
                "policy": exp["policy"],
                "final_accuracy": accuracies[-1] if accuracies else 0,
                "best_accuracy": max(accuracies) if accuracies else 0,
                "total_keys_generated": total_keys,
                "total_bytes_sent": total_bytes,
                "num_rounds": len(rounds),
            })
        
        return comparison

    def close(self):
        """Close the database connection."""
        self.conn.close()

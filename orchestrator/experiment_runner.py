"""
Experiment Runner for QuantumShieldFL

Orchestrates the full federated learning pipeline with three modes:
1. No Encryption — Baseline FL without QKD or AES
2. Fixed QKD — Key regeneration every K rounds
3. Adaptive QKD — Risk-based key management (proposed approach)

Each experiment runs for N rounds with M clients and logs all metrics
for comparison in the dashboard.
"""

import time
import copy
import os
import yaml
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from federated.model import (
    get_model, get_model_params, set_model_params,
    serialize_model_params, deserialize_model_params
)
from federated.client import FLClient
from federated.server import FLServer
from federated.data_loader import get_client_dataloaders
from quantum.bb84 import BB84Protocol
from quantum.key_manager import KeyManager
from crypto.aes_handler import encrypt_model_update, decrypt_model_update
from security.risk_scorer import RiskScorer
from security.adaptive_policy import (
    FixedKeyPolicy, AdaptiveKeyPolicy, NoEncryptionPolicy
)
from attacks.eve_intercept import Eavesdropper
from attacks.model_poisoning import ByzantineAttacker
from metrics.logger import MetricsLogger
from metrics.db import MetricsDB


class ExperimentRunner:
    """
    Runs a complete QuantumShieldFL experiment.
    
    Coordinates all components: FL clients/server, BB84 QKD, AES encryption,
    attack simulation, risk scoring, and metrics logging.
    """

    def __init__(self, experiment_config=None):
        """
        Initialize the experiment runner.
        
        Args:
            experiment_config: dict with experiment parameters, or path to YAML
        """
        # Load config
        if isinstance(experiment_config, str):
            with open(experiment_config, 'r') as f:
                self.exp_config = yaml.safe_load(f)
        elif experiment_config is not None:
            self.exp_config = experiment_config
        else:
            self.exp_config = {}
        
        # Extract parameters (with defaults from config.py)
        self.experiment_name = self.exp_config.get("name", "default")
        self.policy_type = self.exp_config.get("policy", "none")
        self.num_clients = self.exp_config.get("num_clients", config.NUM_CLIENTS)
        self.num_rounds = self.exp_config.get("num_rounds", config.NUM_ROUNDS)
        self.local_epochs = self.exp_config.get("local_epochs", config.LOCAL_EPOCHS)
        self.attack_start = self.exp_config.get("attack_start_round", config.ATTACK_START_ROUND)
        self.attack_end = self.exp_config.get("attack_end_round", config.ATTACK_END_ROUND)
        self.eve_prob = self.exp_config.get("eve_intercept_prob", config.EVE_INTERCEPT_PROBABILITY)
        self.byzantine_clients = self.exp_config.get("byzantine_clients", [])
        self.attack_type = self.exp_config.get("attack_type", "noise")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        
        # Initialize components (lazily in run())
        self.server = None
        self.clients = None
        self.key_manager = None
        self.risk_scorer = None
        self.policy = None
        self.eavesdropper = None
        self.byzantine_attacker = None
        self.logger = None
        self.db = None

    def _setup(self):
        """Initialize all components before running."""
        print(f"\n{'='*60}")
        print(f"  Setting up experiment: {self.experiment_name}")
        print(f"  Policy: {self.policy_type}")
        print(f"  Clients: {self.num_clients}, Rounds: {self.num_rounds}")
        print(f"{'='*60}\n")
        
        # 1. Create FL server
        self.server = FLServer(device=self.device)
        
        # 2. Load data and create clients
        client_loaders, self.test_loader = get_client_dataloaders(
            num_clients=self.num_clients,
            batch_size=config.BATCH_SIZE,
        )
        
        self.clients = {}
        for cid in range(self.num_clients):
            self.clients[cid] = FLClient(
                client_id=cid,
                data_loader=client_loaders[cid],
                device=self.device,
            )
        
        # 3. Key management
        self.key_manager = KeyManager()
        
        # 4. Risk scorer
        self.risk_scorer = RiskScorer()
        
        # 5. Key policy
        if self.policy_type == "none":
            self.policy = NoEncryptionPolicy()
        elif self.policy_type == "fixed":
            interval = self.exp_config.get("key_interval", config.FIXED_KEY_INTERVAL)
            self.policy = FixedKeyPolicy(interval=interval)
        elif self.policy_type == "adaptive":
            self.policy = AdaptiveKeyPolicy()
        else:
            raise ValueError(f"Unknown policy: {self.policy_type}")
        
        # 6. Attack simulation
        self.eavesdropper = Eavesdropper(intercept_probability=self.eve_prob)
        if self.byzantine_clients:
            self.byzantine_attacker = ByzantineAttacker(
                attack_type=self.attack_type,
                noise_scale=config.BYZANTINE_NOISE_SCALE,
            )
        
        # 7. Metrics
        log_file = os.path.join(
            config.RESULTS_DIR,
            f"{self.experiment_name}_metrics.jsonl"
        )
        self.logger = MetricsLogger(
            experiment_name=self.experiment_name,
            log_file=log_file,
        )
        
        # 8. Database
        self.db = MetricsDB()
        self.experiment_id = self.db.create_experiment(
            name=self.experiment_name,
            policy=self.policy_type,
            num_clients=self.num_clients,
            num_rounds=self.num_rounds,
            config_dict=self.exp_config,
        )

    def _run_qkd_for_client(self, client_id, round_number, eve_active=False):
        """
        Run BB84 QKD between a client and the server.
        
        Args:
            client_id: Client ID
            round_number: Current FL round
            eve_active: Whether Eve is intercepting
            
        Returns:
            dict with QKD results
        """
        protocol = BB84Protocol()
        result = protocol.run(
            eve_active=eve_active,
            eve_intercept_prob=self.eve_prob,
        )
        
        if result["success"]:
            self.key_manager.store_key(
                client_id=client_id,
                key=result["shared_key"],
                round_number=round_number,
                qber=result["qber"],
                key_length_bits=result["key_length"],
            )
            self.logger.log_key_event(
                round_number=round_number,
                client_id=client_id,
                event_type="generated",
                qber=result["qber"],
                key_length=result["key_length"],
                reason="BB84 successful",
            )
        else:
            self.key_manager.record_rejection()
            self.logger.log_key_event(
                round_number=round_number,
                client_id=client_id,
                event_type="rejected",
                qber=result["qber"],
                reason=f"QBER too high: {result['qber']:.4f}",
            )
        
        return result

    def _run_round(self, round_number):
        """
        Execute a single federated learning round.
        
        Full workflow:
        1. Distribute global model to all clients
        2. Clients train locally
        3. (If encrypted) Run QKD and encrypt updates
        4. Server aggregates updates
        5. Evaluate global model
        6. Compute risk scores
        7. Log all metrics
        
        Returns:
            dict with round results
        """
        round_start = time.time()
        
        # Determine if attack is active this round
        eve_active = self.attack_start <= round_number <= self.attack_end
        
        global_params = self.server.get_global_params()
        
        # ── Step 1 & 2: Distribute model and train locally ──
        client_updates = []
        client_num_samples = []
        client_metrics_list = []
        client_norms = {}
        total_bytes = 0
        keys_generated_this_round = 0
        round_qber = 0.0
        qber_count = 0
        
        for cid in range(self.num_clients):
            client = self.clients[cid]
            
            # Receive global model
            client.receive_global_model(global_params)
            
            # Train locally
            metrics = client.train(epochs=self.local_epochs)
            client_metrics_list.append(metrics)
            
            # Get model update
            update_params = client.get_model_update()
            update_norm = client.get_update_norm()
            client_norms[cid] = update_norm
            
            # ── Step 3: Byzantine attack (poison update) ──
            is_byzantine = cid in self.byzantine_clients and eve_active
            if is_byzantine and self.byzantine_attacker:
                update_params = self.byzantine_attacker.poison_update(update_params)
                self.logger.log_attack_event(
                    round_number=round_number,
                    attack_type=f"byzantine_{self.attack_type}",
                    detected=False,
                    details={"client_id": cid},
                )
            
            # ── Step 4: QKD + Encryption (if policy requires) ──
            if self.policy_type != "none":
                # Check risk and decide on key regeneration
                risk_result = self.risk_scorer.compute_risk(
                    qber=round_qber / max(1, qber_count) if qber_count > 0 else 0,
                    divergences=[],  # Will be computed after all updates
                    client_norms=client_norms,
                    key_age_rounds=self.key_manager.get_key_age_rounds(
                        cid, round_number
                    ),
                    round_number=round_number,
                )
                
                policy_decision = self.policy.should_regenerate(
                    client_id=cid,
                    current_round=round_number,
                    key_manager=self.key_manager,
                    risk_result=risk_result,
                )
                
                if policy_decision["regenerate"] or not self.key_manager.has_valid_key(cid):
                    # Run QKD
                    qkd_result = self._run_qkd_for_client(
                        client_id=cid,
                        round_number=round_number,
                        eve_active=eve_active,
                    )
                    
                    if qkd_result["success"]:
                        keys_generated_this_round += 1
                    
                    round_qber += qkd_result["qber"]
                    qber_count += 1
                
                # Encrypt update (if we have a valid key)
                key = self.key_manager.get_key(cid)
                if key is not None:
                    serialized = serialize_model_params(
                        self.clients[cid].model
                    )
                    encrypted = encrypt_model_update(serialized, key)
                    total_bytes += encrypted["encrypted_size"]
                    
                    # Decrypt on server side (simulate transmission)
                    decrypted = decrypt_model_update(
                        encrypted["ciphertext"], encrypted["nonce"], key
                    )
                    update_params = deserialize_model_params(decrypted)
                    set_model_params(self.clients[cid].model, update_params)
                    update_params = get_model_params(self.clients[cid].model)
                else:
                    # No valid key — use unencrypted (log warning)
                    serialized = serialize_model_params(
                        self.clients[cid].model
                    )
                    total_bytes += len(serialized)
            else:
                # No encryption — just count bytes
                serialized = serialize_model_params(
                    self.clients[cid].model
                )
                total_bytes += len(serialized)
            
            client_updates.append(update_params)
            client_num_samples.append(client.num_samples)
            
            # Log client metrics
            self.logger.log_client(
                round_number=round_number,
                client_id=cid,
                train_loss=metrics["loss"],
                train_accuracy=metrics["accuracy"],
                update_norm=update_norm,
                is_byzantine=is_byzantine,
                key_regenerated=(
                    self.policy_type != "none" and 
                    policy_decision.get("regenerate", False)
                ) if self.policy_type != "none" else False,
            )
        
        # ── Step 5: Server aggregation ──
        self.server.aggregate(client_updates, client_num_samples)
        
        # ── Step 6: Evaluate global model ──
        eval_result = self.server.evaluate(self.test_loader)
        
        # ── Step 7: Compute divergences (for risk scoring) ──
        divergences = self.server.compute_update_divergence(client_updates)
        
        # ── Step 8: Final risk assessment for this round ──
        avg_qber = round_qber / max(1, qber_count)
        avg_key_age = np.mean([
            self.key_manager.get_key_age_rounds(cid, round_number)
            for cid in range(self.num_clients)
            if self.key_manager.has_valid_key(cid)
        ]) if self.policy_type != "none" and any(
            self.key_manager.has_valid_key(cid) 
            for cid in range(self.num_clients)
        ) else 0
        
        final_risk = self.risk_scorer.compute_risk(
            qber=avg_qber,
            divergences=divergences,
            client_norms=client_norms,
            key_age_rounds=avg_key_age,
            round_number=round_number,
        )
        
        round_time_ms = (time.time() - round_start) * 1000
        
        # ── Step 9: Log round metrics ──
        self.logger.log_round(
            round_number=round_number,
            accuracy=eval_result["accuracy"],
            loss=eval_result["loss"],
            qber=avg_qber if self.policy_type != "none" else None,
            risk_score=final_risk["risk_score"],
            risk_level=final_risk["risk_level"],
            keys_generated=keys_generated_this_round,
            bytes_sent=total_bytes,
            round_time_ms=round_time_ms,
            num_clients=self.num_clients,
            eavesdropper_active=eve_active,
            attack_detected=final_risk["risk_level"] == "HIGH" and eve_active,
            policy_name=self.policy_type,
        )
        
        # Store in database
        self.db.insert_round(self.experiment_id, {
            "round": round_number,
            "accuracy": eval_result["accuracy"],
            "loss": eval_result["loss"],
            "qber": avg_qber if self.policy_type != "none" else None,
            "risk_score": final_risk["risk_score"],
            "risk_level": final_risk["risk_level"],
            "keys_generated": keys_generated_this_round,
            "bytes_sent": total_bytes,
            "round_time_ms": round_time_ms,
            "eavesdropper_active": eve_active,
            "attack_detected": final_risk["risk_level"] == "HIGH" and eve_active,
        })
        
        # Print progress
        attack_str = " [ATTACK]" if eve_active else ""
        risk_str = f" Risk={final_risk['risk_score']:.3f}({final_risk['risk_level']})"
        qkd_str = f" Keys={keys_generated_this_round}" if self.policy_type != "none" else ""
        print(
            f"  Round {round_number:3d}/{self.num_rounds} | "
            f"Acc={eval_result['accuracy']:.4f} | "
            f"Loss={eval_result['loss']:.4f} | "
            f"Time={round_time_ms:.0f}ms"
            f"{qkd_str}{risk_str}{attack_str}"
        )
        
        return {
            "accuracy": eval_result["accuracy"],
            "loss": eval_result["loss"],
            "qber": avg_qber,
            "risk": final_risk,
            "keys_generated": keys_generated_this_round,
            "bytes_sent": total_bytes,
            "round_time_ms": round_time_ms,
        }

    def run(self):
        """
        Execute the complete experiment.
        
        Returns:
            dict with experiment summary
        """
        self._setup()
        
        print(f"\n{'-'*60}")
        print(f"  Starting training: {self.experiment_name}")
        print(f"  Attack window: rounds {self.attack_start}-{self.attack_end}")
        print(f"{'-'*60}\n")
        
        results = []
        for round_num in range(1, self.num_rounds + 1):
            round_result = self._run_round(round_num)
            results.append(round_result)
        
        # Summary
        summary = self.logger.get_summary()
        
        print(f"\n{'='*60}")
        print(f"  Experiment Complete: {self.experiment_name}")
        print(f"  Final Accuracy: {summary['final_accuracy']:.4f}")
        print(f"  Best Accuracy:  {summary['best_accuracy']:.4f}")
        print(f"  Total Keys:     {summary['total_keys_generated']}")
        print(f"  Total Bytes:    {summary['total_bytes_sent']:,}")
        print(f"{'='*60}\n")
        
        self.db.close()
        
        return summary


def run_all_experiments():
    """
    Run all three experimental conditions for comparison.
    
    1. No Encryption (baseline)
    2. Fixed QKD (every 3 rounds)
    3. Adaptive QKD (risk-based)
    """
    experiments = [
        {
            "name": "no_encryption",
            "policy": "none",
            "num_clients": config.NUM_CLIENTS,
            "num_rounds": config.NUM_ROUNDS,
            "attack_start_round": config.ATTACK_START_ROUND,
            "attack_end_round": config.ATTACK_END_ROUND,
            "byzantine_clients": [3],
            "attack_type": "noise",
        },
        {
            "name": "fixed_qkd",
            "policy": "fixed",
            "num_clients": config.NUM_CLIENTS,
            "num_rounds": config.NUM_ROUNDS,
            "key_interval": config.FIXED_KEY_INTERVAL,
            "attack_start_round": config.ATTACK_START_ROUND,
            "attack_end_round": config.ATTACK_END_ROUND,
            "eve_intercept_prob": config.EVE_INTERCEPT_PROBABILITY,
            "byzantine_clients": [3],
            "attack_type": "noise",
        },
        {
            "name": "adaptive_qkd",
            "policy": "adaptive",
            "num_clients": config.NUM_CLIENTS,
            "num_rounds": config.NUM_ROUNDS,
            "attack_start_round": config.ATTACK_START_ROUND,
            "attack_end_round": config.ATTACK_END_ROUND,
            "eve_intercept_prob": config.EVE_INTERCEPT_PROBABILITY,
            "byzantine_clients": [3],
            "attack_type": "noise",
        },
    ]
    
    summaries = {}
    for exp_config in experiments:
        runner = ExperimentRunner(exp_config)
        summary = runner.run()
        summaries[exp_config["name"]] = summary
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'No Encrypt':<15} {'Fixed QKD':<15} {'Adaptive QKD':<15}")
    print(f"  {'-'*65}")
    
    for metric in ["final_accuracy", "best_accuracy", "total_keys_generated", 
                    "total_bytes_sent", "avg_round_time_ms"]:
        vals = []
        for name in ["no_encryption", "fixed_qkd", "adaptive_qkd"]:
            v = summaries[name].get(metric, 0)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(f"{v:,}")
        print(f"  {metric:<25} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15}")
    
    print(f"{'='*70}\n")
    
    return summaries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QuantumShieldFL Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to experiment YAML config")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    args = parser.parse_args()
    
    if args.all:
        run_all_experiments()
    elif args.config:
        runner = ExperimentRunner(args.config)
        runner.run()
    else:
        # Default: run all experiments
        run_all_experiments()

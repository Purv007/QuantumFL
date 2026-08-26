"""
QuantumShieldFL — Global Configuration

All hyperparameters and settings for the federated learning pipeline,
quantum key distribution, encryption, risk assessment, and experiments.
"""

import os

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments", "results")
DB_PATH = os.path.join(PROJECT_ROOT, "experiments", "results", "metrics.db")

# ─────────────────────────────────────────────
# Federated Learning
# ─────────────────────────────────────────────
NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.01
MOMENTUM = 0.9
NON_IID = True                 # Use non-IID data partitioning
NON_IID_ALPHA = 0.5            # Dirichlet alpha (lower = more skewed)

# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
MODEL_NAME = "SimpleCNN"       # Model architecture identifier
NUM_CLASSES = 10               # MNIST has 10 digit classes
INPUT_CHANNELS = 1             # Grayscale images

# ─────────────────────────────────────────────
# Quantum Key Distribution (BB84)
# ─────────────────────────────────────────────
QKD_NUM_QUBITS = 1024          # Number of qubits per BB84 session
QKD_SAMPLE_FRACTION = 0.25     # Fraction of sifted bits used for QBER estimation
QBER_THRESHOLD = 0.11          # Max acceptable QBER (theoretical: 11%)
MIN_KEY_LENGTH = 256           # Minimum key length in bits for AES-256

# ─────────────────────────────────────────────
# Encryption (AES-256-GCM)
# ─────────────────────────────────────────────
AES_KEY_SIZE = 32              # 256 bits = 32 bytes
AES_NONCE_SIZE = 12            # GCM standard nonce size

# ─────────────────────────────────────────────
# Attack Simulation
# ─────────────────────────────────────────────
EVE_INTERCEPT_PROBABILITY = 1.0    # Probability Eve intercepts each qubit
ATTACK_START_ROUND = 8             # Round when attack begins
ATTACK_END_ROUND = 14              # Round when attack ends
BYZANTINE_NUM_ATTACKERS = 1        # Number of malicious clients
BYZANTINE_NOISE_SCALE = 5.0        # Scale of noise injected by attacker

# ─────────────────────────────────────────────
# Risk Assessment & Adaptive Key Management
# ─────────────────────────────────────────────
RISK_LOW_THRESHOLD = 0.3           # Below this → low risk
RISK_HIGH_THRESHOLD = 0.7          # Above this → high risk
MAX_KEY_AGE = 5                    # Max rounds a key can be reused

# Risk score component weights (must sum to 1.0)
RISK_WEIGHT_QBER = 0.30
RISK_WEIGHT_DIVERGENCE = 0.25
RISK_WEIGHT_ANOMALY = 0.20
RISK_WEIGHT_KEY_AGE = 0.15
RISK_WEIGHT_TREND = 0.10

# Fixed-key policy: regenerate every K rounds
FIXED_KEY_INTERVAL = 3

# ─────────────────────────────────────────────
# Metrics & Logging
# ─────────────────────────────────────────────
LOG_LEVEL = "INFO"
METRICS_LOG_FILE = os.path.join(RESULTS_DIR, "metrics.jsonl")

# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────
DASHBOARD_PORT = 8501
DASHBOARD_THEME = "dark"

# ─────────────────────────────────────────────
# Random Seed (for reproducibility)
# ─────────────────────────────────────────────
RANDOM_SEED = 42

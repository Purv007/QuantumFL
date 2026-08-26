<div align="center">

# 🔐 QuantumShieldFL

**An Adaptive Privacy-Preserving Federated Learning Framework  
Secured by Quantum Key Distribution**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4?style=for-the-badge&logo=ibm&logoColor=white)](https://qiskit.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

**QuantumShieldFL** is a research prototype that fuses **Federated Learning (FL)**, **Quantum Key Distribution (QKD)**, and **Adaptive Risk-Based Security** into a unified, closed-loop architecture for privacy-preserving machine learning.

Unlike traditional FL systems where cryptography and model training operate as independent silos, QuantumShieldFL creates a **bi-directional feedback loop** — the AI's anomaly detection drives quantum key management, while the quantum channel's physical error rates inform the AI's trust scoring. This cross-domain fusion is the core novelty of the framework.

> **Note:** The quantum component is a **simulation** using IBM's Qiskit Aer backend. No real quantum hardware is required.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Federated Learning** | 5 clients collaboratively train a CNN on MNIST without sharing raw data |
| **BB84 QKD Simulation** | Quantum key generation using Qiskit's `AerSimulator` with full circuit simulation |
| **AES-256-GCM Encryption** | Model updates encrypted with quantum-derived symmetric keys before transmission |
| **Eavesdropper Detection** | BB84 protocol detects interception via Quantum Bit Error Rate (QBER) analysis |
| **Adaptive Key Management** | Risk-based key regeneration that dynamically responds to threat conditions |
| **Attack Simulation** | Eve's intercept-and-resend attack on QKD + Byzantine model poisoning on FL |
| **Interactive Dashboard** | Real-time Streamlit dashboard with experiment comparison and playground |

---

## 🏗️ System Architecture

```
                          ┌─────────────────────────┐
                          │    Central FL Server     │
                          │   (FedAvg Aggregation)   │
                          └────┬──────────────┬──────┘
                               │              │
                    ┌──────────▼──┐      ┌────▼────────────┐
                    │  Risk Scorer │◄────►│  Adaptive Policy │
                    │ (Anomaly AI) │      │ (Key Management) │
                    └──────┬───┬──┘      └────────┬────────┘
                           │   │                  │
              ┌────────────┘   └──────┐           │
              ▼                       ▼           ▼
     ┌────────────────┐     ┌─────────────┐  ┌────────────┐
     │ Classical Metrics│    │ QBER Metrics │  │ BB84 QKD   │
     │ (Loss, Diverge) │    │ (Quantum)    │  │ (Qiskit)   │
     └────────────────┘     └─────────────┘  └──────┬─────┘
                                                     │
                                              ┌──────▼──────┐
              ┌──────────┐                    │ AES-256-GCM  │
  Client 1 ──┤          ├── [Encrypted] ────►│  Encryption  │
  Client 2 ──┤ Local    ├── [Model     ] ────►│              │
  Client 3 ──┤ Training ├── [Updates   ] ────►│              │
  Client 4 ──┤ (CNN)    ├── [         ] ────►│              │
  Client 5 ──┤          ├── [         ] ────►│              │
              └──────────┘                    └──────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantumShieldFL.git
cd QuantumShieldFL

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Run all three benchmark experiments:**
```bash
python -m orchestrator.experiment_runner --all
```

**Launch the interactive dashboard:**
```bash
streamlit run dashboard/app.py
```

**Run the test suite:**
```bash
pytest tests/ -v
```

---

## 🧪 Experimental Evaluation

The system benchmarks three experimental conditions across 20 federated training rounds:

| Condition | Encryption | Eve Detection | Attack Response | Key Overhead |
|---|---|---|---|---|
| **No Encryption** | ❌ None | ❌ None | ❌ None | None |
| **Fixed QKD** | ✅ AES-256 | ✅ QBER | ⚠️ Periodic | Fixed (every N rounds) |
| **Adaptive QKD** *(proposed)* | ✅ AES-256 | ✅ QBER | ✅ Dynamic | Minimized (risk-driven) |

### Experiment Configs

Each experiment is defined by a YAML configuration file in `experiments/configs/`:

| Config File | Description |
|---|---|
| `no_encryption.yaml` | Baseline — plain FL with no security layer |
| `fixed_qkd.yaml` | QKD keys regenerated on a fixed schedule (every 3 rounds) |
| `adaptive_qkd.yaml` | QKD keys regenerated only when the Risk Scorer detects anomalies |

Run a single experiment:
```bash
python -m orchestrator.experiment_runner --config experiments/configs/adaptive_qkd.yaml
```

---

## 📂 Project Structure

```
QuantumShieldFL/
│
├── config.py                        # Global configuration & hyperparameters
├── requirements.txt                 # Python dependencies
│
├── federated/                       # ── Federated Learning Engine ──
│   ├── model.py                     #    CNN architecture (MNIST)
│   ├── data_loader.py               #    IID / non-IID data partitioning
│   ├── client.py                    #    Local client training logic
│   └── server.py                    #    FedAvg server aggregation
│
├── quantum/                         # ── Quantum Key Distribution ──
│   ├── bb84.py                      #    BB84 protocol (Qiskit AerSimulator)
│   └── key_manager.py               #    Key lifecycle & rotation management
│
├── crypto/                          # ── Cryptographic Layer ──
│   └── aes_handler.py               #    AES-256-GCM encrypt / decrypt
│
├── security/                        # ── Adaptive Security ──
│   ├── risk_scorer.py               #    Multi-signal risk computation
│   └── adaptive_policy.py           #    Dynamic vs fixed key policies
│
├── attacks/                         # ── Attack Simulation ──
│   ├── eve_intercept.py             #    QKD eavesdropping (intercept-resend)
│   └── model_poisoning.py           #    Byzantine FL model poisoning
│
├── orchestrator/                    # ── Experiment Coordination ──
│   └── experiment_runner.py         #    Multi-experiment runner & comparator
│
├── dashboard/                       # ── Visualization ──
│   └── app.py                       #    Streamlit interactive dashboard
│
├── metrics/                         # ── Logging & Persistence ──
│   ├── logger.py                    #    JSONL structured event logging
│   └── db.py                        #    SQLite database interface
│
├── experiments/                     # ── Configs & Results ──
│   ├── configs/                     #    YAML experiment configurations
│   └── results/                     #    Generated output data
│
└── tests/                           # ── Test Suite ──
    ├── test_bb84.py                 #    QKD protocol tests
    ├── test_aes.py                  #    Encryption round-trip tests
    ├── test_federated.py            #    FL training & aggregation tests
    ├── test_risk_scorer.py          #    Risk scoring logic tests
    └── test_integration.py          #    End-to-end integration tests
```

---

## 🛡️ Security Model

### Threat Model

| Threat | Mitigation |
|---|---|
| **Eavesdropping** (Man-in-the-Middle) | BB84 QKD detects interception via QBER; compromised keys are discarded |
| **Model Poisoning** (Byzantine Clients) | Risk Scorer detects anomalous weight updates via statistical divergence |
| **Replay Attacks** | AES-GCM authenticated encryption with unique nonces per message |
| **Key Compromise** | Adaptive policy triggers immediate quantum key regeneration |

### BB84 Protocol Flow

```
Alice (Server)              Quantum Channel              Bob (Client)
     │                            │                            │
     │── Prepare qubits ─────────►│                            │
     │   (random bits + bases)    │                            │
     │                            │◄── Eve intercepts? ──────►│
     │                            │    (causes QBER spike)     │
     │                            │────── Measure qubits ─────►│
     │                            │       (random bases)       │
     │◄── Basis comparison ───────┤──── Basis comparison ─────►│
     │                            │                            │
     │── Sifted key ──────────────┤──── Sifted key ───────────►│
     │                            │                            │
     │── QBER check ──────────────┤──── QBER check ──────────►│
     │   (if QBER > 20%: ABORT)   │                            │
```

---

## 🖥️ Dashboard

The Streamlit dashboard provides five interactive tabs:

| Tab | Purpose |
|---|---|
| **📊 Overview** | High-level metrics: accuracy trends, QBER history, risk scores |
| **📈 Training Progress** | Per-round FL metrics with convergence analysis |
| **🔬 Experiment Comparison** | Side-by-side comparison of all three experimental conditions |
| **🎮 Interactive Playground** | Live BB84 simulation, risk scoring, and attack testing |
| **🗄️ Database Explorer** | Browse all raw experiment data stored in SQLite |

Launch with:
```bash
streamlit run dashboard/app.py
```

---

## 🧰 Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Neural Networks | PyTorch 2.0+ | CNN training and federated aggregation |
| Quantum Simulation | Qiskit 1.0+ / Aer | BB84 quantum circuit simulation |
| Encryption | cryptography (pyca) | AES-256-GCM authenticated encryption |
| Dashboard | Streamlit + Plotly | Interactive visualization and monitoring |
| Data Persistence | SQLite | Experiment metrics and event logging |
| Testing | pytest | Unit and integration test suite |
| ML Utilities | scikit-learn, NumPy, pandas | Data processing and analysis |

---

## 🧪 Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_bb84.py -v

# Run with coverage report
pytest tests/ -v --tb=short
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [IBM Qiskit](https://qiskit.org/) — Quantum computing framework
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Streamlit](https://streamlit.io/) — Dashboard framework
- BB84 Protocol — C.H. Bennett & G. Brassard (1984)
- FedAvg Algorithm — McMahan et al. (2017)

---

<div align="center">

**Built with ⚛️ Quantum Computing and 🤖 Federated Learning**

</div>

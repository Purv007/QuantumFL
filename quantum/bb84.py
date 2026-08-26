"""
BB84 Quantum Key Distribution Protocol Simulation

Implements the BB84 protocol using Qiskit's quantum circuit simulator.
The protocol allows two parties (Alice and Bob) to generate a shared
secret key while detecting any eavesdropping attempts.

Protocol Steps:
1. Alice prepares qubits in random bases (rectilinear |0⟩/|1⟩ or diagonal |+⟩/|-⟩)
2. Alice encodes random bits into the chosen bases
3. Bob measures each qubit in a randomly chosen basis
4. They publicly compare bases (sifting) and keep only matching-basis bits
5. They sample a subset to estimate the Quantum Bit Error Rate (QBER)
6. If QBER < threshold → remaining bits become the shared secret key
7. If QBER ≥ threshold → eavesdropper detected, abort key exchange
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class BB84Protocol:
    """
    Simulates the BB84 Quantum Key Distribution protocol.
    
    Uses Qiskit's AerSimulator to simulate quantum measurements.
    The protocol generates shared secret keys between two parties
    while providing the ability to detect eavesdroppers.
    """

    def __init__(self, num_qubits=None, sample_fraction=None, qber_threshold=None):
        """
        Initialize BB84 protocol parameters.
        
        Args:
            num_qubits: Number of qubits to use per session
            sample_fraction: Fraction of sifted bits used for QBER estimation
            qber_threshold: Maximum acceptable QBER before rejecting
        """
        self.num_qubits = num_qubits or config.QKD_NUM_QUBITS
        self.sample_fraction = sample_fraction or config.QKD_SAMPLE_FRACTION
        self.qber_threshold = qber_threshold or config.QBER_THRESHOLD
        self.simulator = AerSimulator()
        
        # Protocol state (populated after run)
        self.alice_bits = None
        self.alice_bases = None
        self.bob_bases = None
        self.bob_results = None
        self.sifted_alice_bits = None
        self.sifted_bob_bits = None
        self.qber = None
        self.shared_key = None
        self.key_length = 0
        self.eavesdropper_detected = False
        self.protocol_success = False

    def _prepare_alice(self):
        """
        Alice's preparation phase.
        
        Alice generates random bits and random bases:
        - Bit 0 or 1
        - Base 0 (rectilinear/Z-basis: |0⟩, |1⟩) or 1 (diagonal/X-basis: |+⟩, |-⟩)
        """
        self.alice_bits = np.random.randint(0, 2, size=self.num_qubits)
        self.alice_bases = np.random.randint(0, 2, size=self.num_qubits)

    def _prepare_bob(self):
        """
        Bob's measurement preparation.
        
        Bob independently chooses random measurement bases.
        """
        self.bob_bases = np.random.randint(0, 2, size=self.num_qubits)

    def _create_quantum_circuit(self, qubit_idx, alice_bit, alice_base, 
                                 bob_base, eve_intercept=False, eve_base=None):
        """
        Create a quantum circuit for a single qubit in the BB84 protocol.
        
        Args:
            qubit_idx: Index of the qubit
            alice_bit: Alice's bit value (0 or 1)
            alice_base: Alice's basis choice (0=Z, 1=X)
            bob_base: Bob's basis choice (0=Z, 1=X)
            eve_intercept: Whether Eve intercepts this qubit
            eve_base: Eve's measurement basis (if intercepting)
            
        Returns:
            QuantumCircuit for this qubit
        """
        qc = QuantumCircuit(1, 1)
        
        # Step 1: Alice encodes her bit
        if alice_bit == 1:
            qc.x(0)  # Flip to |1⟩
        
        # Step 2: If Alice uses diagonal basis, apply Hadamard
        if alice_base == 1:
            qc.h(0)  # |0⟩ → |+⟩, |1⟩ → |-⟩
        
        # Step 3: Eve's intercept-and-resend (if active)
        if eve_intercept and eve_base is not None:
            # Eve measures in her chosen basis
            if eve_base == 1:
                qc.h(0)
            qc.measure(0, 0)
            # Eve resends — we simulate this by preparing a new state
            # based on Eve's measurement result (handled in simulation)
            # For simulation, we just add the measurement which collapses state
            qc.h(0) if eve_base == 1 else None
            # Reset and re-prepare based on Eve's measurement
            # (In simulation, the measure already collapsed the state)
            if eve_base == 1:
                qc.h(0)
        
        # Step 4: Bob measures in his chosen basis
        if bob_base == 1:
            qc.h(0)  # Rotate to diagonal basis before measurement
        
        qc.measure(0, 0)
        
        return qc

    def _simulate_qubit_exchange(self, eve_active=False, eve_intercept_prob=1.0):
        """
        Simulate the quantum channel for all qubits.
        
        Runs each qubit through the BB84 circuit and collects Bob's results.
        Optionally includes Eve's intercept-and-resend attack.
        
        Args:
            eve_active: Whether Eve is present on the channel
            eve_intercept_prob: Probability Eve intercepts each qubit
            
        Returns:
            numpy array of Bob's measurement results
        """
        bob_results = np.zeros(self.num_qubits, dtype=int)
        eve_bases = None
        
        if eve_active:
            eve_bases = np.random.randint(0, 2, size=self.num_qubits)
        
        for i in range(self.num_qubits):
            qc = QuantumCircuit(1, 1)
            
            # Alice prepares qubit
            if self.alice_bits[i] == 1:
                qc.x(0)
            if self.alice_bases[i] == 1:
                qc.h(0)
            
            # Eve intercepts (if active and coin flip allows)
            if eve_active and np.random.random() < eve_intercept_prob:
                # Eve measures in her random basis
                if eve_bases[i] == 1:
                    qc.h(0)
                qc.measure(0, 0)
                
                # Eve must resend to Bob — create new circuit continuing from collapse
                # We simulate by running Eve's measurement first
                eve_result = self.simulator.run(qc, shots=1).result()
                eve_measured_bit = int(list(eve_result.get_counts().keys())[0])
                
                # Eve re-prepares qubit based on her measurement result and basis
                qc2 = QuantumCircuit(1, 1)
                if eve_measured_bit == 1:
                    qc2.x(0)
                if eve_bases[i] == 1:
                    qc2.h(0)
                
                # Bob measures
                if self.bob_bases[i] == 1:
                    qc2.h(0)
                qc2.measure(0, 0)
                
                result = self.simulator.run(qc2, shots=1).result()
                bob_results[i] = int(list(result.get_counts().keys())[0])
            else:
                # No Eve — Bob measures directly
                if self.bob_bases[i] == 1:
                    qc.h(0)
                qc.measure(0, 0)
                
                result = self.simulator.run(qc, shots=1).result()
                bob_results[i] = int(list(result.get_counts().keys())[0])
        
        return bob_results

    def _sift_keys(self):
        """
        Sifting phase: keep only bits where Alice and Bob used the same basis.
        
        On average, ~50% of bits survive sifting.
        """
        matching_indices = np.where(self.alice_bases == self.bob_bases)[0]
        
        self.sifted_alice_bits = self.alice_bits[matching_indices]
        self.sifted_bob_bits = self.bob_results[matching_indices]
        
        return matching_indices

    def _estimate_qber(self):
        """
        Estimate the Quantum Bit Error Rate by comparing a random sample.
        
        QBER = (number of mismatched bits in sample) / (sample size)
        
        Without eavesdropping: QBER ≈ 0%
        With full intercept-resend: QBER ≈ 25%
        Threshold for security: typically 11%
        
        Returns:
            float: Estimated QBER
        """
        num_sifted = len(self.sifted_alice_bits)
        sample_size = max(1, int(num_sifted * self.sample_fraction))
        
        # Randomly select bits for error estimation
        sample_indices = np.random.choice(num_sifted, size=sample_size, replace=False)
        
        # Compare Alice's and Bob's bits at sample positions
        alice_sample = self.sifted_alice_bits[sample_indices]
        bob_sample = self.sifted_bob_bits[sample_indices]
        
        errors = np.sum(alice_sample != bob_sample)
        self.qber = errors / sample_size
        
        # Remove sampled bits from the key (they've been publicly revealed)
        remaining_mask = np.ones(num_sifted, dtype=bool)
        remaining_mask[sample_indices] = False
        
        self.sifted_alice_bits = self.sifted_alice_bits[remaining_mask]
        self.sifted_bob_bits = self.sifted_bob_bits[remaining_mask]
        
        return self.qber

    def _generate_key(self):
        """
        Generate the shared secret key from remaining sifted bits.
        
        Converts bit array to bytes for use as AES encryption key.
        
        Returns:
            tuple: (key_bytes, key_length_bits)
        """
        # Use Alice's remaining sifted bits as the key
        key_bits = self.sifted_alice_bits
        self.key_length = len(key_bits)
        
        # Convert bits to bytes (pad to multiple of 8)
        num_bytes = self.key_length // 8
        if num_bytes == 0:
            return None, 0
        
        # Take only complete bytes worth of bits
        usable_bits = key_bits[:num_bytes * 8]
        
        key_bytes = bytearray()
        for i in range(0, len(usable_bits), 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | int(usable_bits[i + j])
            key_bytes.append(byte_val)
        
        self.shared_key = bytes(key_bytes)
        return self.shared_key, self.key_length

    def run(self, eve_active=False, eve_intercept_prob=1.0):
        """
        Execute the full BB84 protocol.
        
        Args:
            eve_active: Whether an eavesdropper is present
            eve_intercept_prob: Probability of interception per qubit
            
        Returns:
            dict with protocol results:
                - success: bool — whether key exchange succeeded
                - shared_key: bytes or None — the generated key
                - qber: float — quantum bit error rate
                - key_length: int — key length in bits
                - eavesdropper_detected: bool
                - num_qubits: int — qubits used
                - sifted_bits: int — bits after sifting
        """
        # Phase 1: Preparation
        self._prepare_alice()
        self._prepare_bob()
        
        # Phase 2: Quantum channel (with optional Eve)
        self.bob_results = self._simulate_qubit_exchange(
            eve_active=eve_active,
            eve_intercept_prob=eve_intercept_prob
        )
        
        # Phase 3: Sifting
        self._sift_keys()
        num_sifted = len(self.sifted_alice_bits)
        
        # Phase 4: QBER estimation
        qber = self._estimate_qber()
        
        # Phase 5: Security decision
        self.eavesdropper_detected = qber >= self.qber_threshold
        
        if self.eavesdropper_detected:
            self.protocol_success = False
            self.shared_key = None
            self.key_length = 0
            print(f"    [!] BB84: Eavesdropper detected! QBER={qber:.4f} "
                  f"(threshold={self.qber_threshold})")
        else:
            # Phase 6: Key generation
            key, key_len = self._generate_key()
            
            if key is None or len(key) < config.MIN_KEY_LENGTH // 8:
                self.protocol_success = False
                print(f"    [!] BB84: Key too short ({self.key_length} bits). "
                      f"Need {config.MIN_KEY_LENGTH} bits.")
            else:
                self.protocol_success = True
                # Truncate or hash key to exactly AES_KEY_SIZE bytes
                if len(self.shared_key) >= config.AES_KEY_SIZE:
                    self.shared_key = self.shared_key[:config.AES_KEY_SIZE]
                else:
                    # Stretch key using SHA-256 if too short
                    import hashlib
                    self.shared_key = hashlib.sha256(self.shared_key).digest()
                
                print(f"    [OK] BB84: Key generated successfully. "
                      f"QBER={qber:.4f}, Key={len(self.shared_key)*8} bits")
        
        return {
            "success": self.protocol_success,
            "shared_key": self.shared_key,
            "qber": qber,
            "key_length": len(self.shared_key) * 8 if self.shared_key else 0,
            "eavesdropper_detected": self.eavesdropper_detected,
            "num_qubits": self.num_qubits,
            "sifted_bits": num_sifted,
        }


def run_bb84(num_qubits=None, eve_active=False, eve_intercept_prob=1.0):
    """
    Convenience function to run a single BB84 session.
    
    Args:
        num_qubits: Number of qubits for the session
        eve_active: Whether eavesdropper is present
        eve_intercept_prob: Eavesdropper interception probability
        
    Returns:
        dict with protocol results
    """
    protocol = BB84Protocol(num_qubits=num_qubits)
    return protocol.run(
        eve_active=eve_active, 
        eve_intercept_prob=eve_intercept_prob
    )

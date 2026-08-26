"""
Tests for AES-256-GCM encryption handler.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from cryptography.exceptions import InvalidTag
from crypto.aes_handler import AESCipher, encrypt_model_update, decrypt_model_update


class TestAESCipher:
    """Test suite for AES-256-GCM encryption."""

    def setup_method(self):
        """Create a test key and cipher."""
        self.key = os.urandom(32)  # 256-bit key
        self.cipher = AESCipher(self.key)
        self.plaintext = b"Hello, QuantumShieldFL! This is a test message."

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting then decrypting should return original plaintext."""
        result = self.cipher.encrypt(self.plaintext)
        decrypted = self.cipher.decrypt(result["ciphertext"], result["nonce"])
        
        assert decrypted == self.plaintext

    def test_wrong_key_fails(self):
        """Decryption with wrong key should raise InvalidTag."""
        result = self.cipher.encrypt(self.plaintext)
        wrong_key = os.urandom(32)
        wrong_cipher = AESCipher(wrong_key)
        
        with pytest.raises(InvalidTag):
            wrong_cipher.decrypt(result["ciphertext"], result["nonce"])

    def test_tampered_ciphertext_fails(self):
        """Modified ciphertext should be detected by GCM auth tag."""
        result = self.cipher.encrypt(self.plaintext)
        
        # Tamper with ciphertext
        tampered = bytearray(result["ciphertext"])
        tampered[0] ^= 0xFF
        
        with pytest.raises(InvalidTag):
            self.cipher.decrypt(bytes(tampered), result["nonce"])

    def test_encrypted_size_larger(self):
        """Ciphertext should be larger than plaintext (GCM tag overhead)."""
        result = self.cipher.encrypt(self.plaintext)
        
        # GCM adds 16-byte auth tag
        assert result["encrypted_size"] == len(self.plaintext) + 16

    def test_encrypt_model_update_convenience(self):
        """Test the convenience functions for model update encryption."""
        serialized = b"fake_model_params_bytes_" * 100
        
        encrypted = encrypt_model_update(serialized, self.key)
        decrypted = decrypt_model_update(
            encrypted["ciphertext"], encrypted["nonce"], self.key
        )
        
        assert decrypted == serialized
        assert encrypted["original_size"] == len(serialized)

    def test_different_nonces(self):
        """Each encryption should use a different nonce."""
        r1 = self.cipher.encrypt(self.plaintext)
        r2 = self.cipher.encrypt(self.plaintext)
        
        assert r1["nonce"] != r2["nonce"]
        assert r1["ciphertext"] != r2["ciphertext"]

    def test_key_size_validation(self):
        """Invalid key sizes should raise AssertionError."""
        with pytest.raises(AssertionError):
            AESCipher(b"short_key")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

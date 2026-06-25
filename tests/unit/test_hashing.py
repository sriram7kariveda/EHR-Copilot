"""Unit tests for hashing utilities (utils/hashing.py)."""

from __future__ import annotations

import pytest

from ehr_copilot.utils.hashing import (
    compute_chain_hash,
    sha256_hex,
    verify_chain,
)


class TestSha256Hex:
    def test_deterministic(self):
        """Same input always yields the same hash."""
        result_a = sha256_hex("hello world")
        result_b = sha256_hex("hello world")
        assert result_a == result_b

    def test_hex_length(self):
        """SHA-256 hex digest is always 64 characters."""
        result = sha256_hex("test data")
        assert len(result) == 64

    def test_different_inputs_differ(self):
        assert sha256_hex("aaa") != sha256_hex("bbb")


class TestComputeChainHash:
    def test_chain_hash_combines_inputs(self):
        prev = sha256_hex("genesis")
        entry_data = "some entry payload"
        chain_hash = compute_chain_hash(prev, entry_data)
        assert len(chain_hash) == 64
        # Should equal sha256_hex(prev + entry_data)
        expected = sha256_hex(prev + entry_data)
        assert chain_hash == expected

    def test_chain_hash_deterministic(self):
        h1 = compute_chain_hash("prev", "data")
        h2 = compute_chain_hash("prev", "data")
        assert h1 == h2

    def test_chain_hash_sensitive_to_order(self):
        h1 = compute_chain_hash("A", "B")
        h2 = compute_chain_hash("B", "A")
        assert h1 != h2


class TestVerifyChain:
    def _build_chain(self, payloads: list[str]) -> list[tuple[str, str, str]]:
        """Build a valid chain of (previous_hash, data, entry_hash) tuples."""
        chain: list[tuple[str, str, str]] = []
        prev_hash = "0"  # Seed
        for payload in payloads:
            entry_hash = compute_chain_hash(prev_hash, payload)
            chain.append((prev_hash, payload, entry_hash))
            prev_hash = entry_hash
        return chain

    def test_valid_chain(self):
        chain = self._build_chain(["entry1", "entry2", "entry3"])
        assert verify_chain(chain) is True

    def test_empty_chain(self):
        assert verify_chain([]) is True

    def test_single_entry(self):
        chain = self._build_chain(["only_entry"])
        assert verify_chain(chain) is True

    def test_tampered_hash_detected(self):
        chain = self._build_chain(["entry1", "entry2", "entry3"])
        # Tamper with the hash of the second entry
        prev, data, _hash = chain[1]
        chain[1] = (prev, data, "0000000000000000000000000000000000000000000000000000000000000000")
        assert verify_chain(chain) is False

    def test_tampered_data_detected(self):
        chain = self._build_chain(["entry1", "entry2", "entry3"])
        # Modify the data of the second entry while keeping the old hash
        prev, _data, entry_hash = chain[1]
        chain[1] = (prev, "TAMPERED DATA", entry_hash)
        assert verify_chain(chain) is False

    def test_broken_link_detected(self):
        chain = self._build_chain(["entry1", "entry2", "entry3"])
        # Break the linkage: make entry2's previous_hash not match entry1's hash
        _prev, data, entry_hash = chain[1]
        wrong_prev = "aaaa" * 16  # 64 chars, but wrong
        new_hash = compute_chain_hash(wrong_prev, data)
        chain[1] = (wrong_prev, data, new_hash)
        assert verify_chain(chain) is False

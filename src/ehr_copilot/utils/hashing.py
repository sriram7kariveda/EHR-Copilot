"""SHA-256 utilities for the audit chain.

Provides functions to compute hex digests, build chained hashes for audit
log entries, and verify the integrity of an entire chain.
"""

from __future__ import annotations

import hashlib


def sha256_hex(data: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 encoded string.

    Parameters
    ----------
    data:
        The input string to hash.

    Returns
    -------
    str
        Lowercase hex-encoded SHA-256 digest (64 characters).
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_chain_hash(previous_hash: str, entry_data: str) -> str:
    """Compute the hash for an audit-chain entry.

    The hash is ``SHA-256(previous_hash + entry_data)`` where ``+`` denotes
    simple string concatenation.

    Parameters
    ----------
    previous_hash:
        The hex digest of the preceding entry in the chain.  For the very
        first entry this is typically a known seed value (e.g. ``"0"``).
    entry_data:
        The serialised payload of the current audit entry.

    Returns
    -------
    str
        Lowercase hex-encoded SHA-256 digest for this entry.
    """
    return sha256_hex(previous_hash + entry_data)


def verify_chain(entries: list[tuple[str, str, str]]) -> bool:
    """Verify the integrity of an audit chain.

    Each element of *entries* is a 3-tuple
    ``(previous_hash, data, entry_hash)`` where ``entry_hash`` must equal
    ``SHA-256(previous_hash + data)``.

    Additionally, for every pair of consecutive entries the
    ``previous_hash`` field of entry *i+1* must equal the ``entry_hash``
    of entry *i*.

    Parameters
    ----------
    entries:
        Ordered sequence of ``(previous_hash, data, entry_hash)`` tuples
        representing the audit chain from oldest to newest.

    Returns
    -------
    bool
        ``True`` if every entry hash is correct **and** the chain links
        are consistent; ``False`` otherwise.
    """
    if not entries:
        return True

    for idx, (previous_hash, data, entry_hash) in enumerate(entries):
        # 1. The stored hash must match the recomputed hash.
        expected_hash = compute_chain_hash(previous_hash, data)
        if entry_hash != expected_hash:
            return False

        # 2. The previous_hash of the *next* entry must equal this
        #    entry's hash (chain linkage).
        if idx > 0:
            _, _, prev_entry_hash = entries[idx - 1]
            if previous_hash != prev_entry_hash:
                return False

    return True

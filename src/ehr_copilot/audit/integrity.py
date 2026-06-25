"""Hash chain verification utilities."""

from __future__ import annotations

import hashlib

from ehr_copilot.domain.audit import AuditEntry
from ehr_copilot.audit.schemas import entry_to_hashable_string


def compute_entry_hash(previous_hash: str, entry_data: str) -> str:
    """Compute SHA-256 hash of (previous_hash + entry_data)."""
    combined = previous_hash + entry_data
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def verify_hash_chain(entries: list[AuditEntry]) -> tuple[bool, list[str]]:
    """Verify the integrity of a sequence of audit entries.

    Checks that:
    - Each entry's hash matches the recomputed hash from its content.
    - Each entry's previous_hash matches the prior entry's entry_hash.

    Args:
        entries: Ordered list of AuditEntry objects forming a hash chain.

    Returns:
        A tuple of (is_valid, error_messages). is_valid is True when the
        entire chain is consistent; error_messages lists any problems found.
    """
    errors: list[str] = []

    for i, entry in enumerate(entries):
        # Recompute the hash from the entry content
        entry_data = entry_to_hashable_string(entry)
        expected_hash = compute_entry_hash(entry.previous_hash, entry_data)

        if entry.entry_hash != expected_hash:
            errors.append(
                f"Entry {entry.entry_id} at index {i}: hash mismatch "
                f"(expected {expected_hash}, got {entry.entry_hash})"
            )

        # Verify the previous_hash linkage
        if i == 0:
            if entry.previous_hash != "":
                errors.append(
                    f"Entry {entry.entry_id} at index 0: expected empty "
                    f"previous_hash for first entry, got '{entry.previous_hash}'"
                )
        else:
            expected_previous = entries[i - 1].entry_hash
            if entry.previous_hash != expected_previous:
                errors.append(
                    f"Entry {entry.entry_id} at index {i}: previous_hash mismatch "
                    f"(expected {expected_previous}, got {entry.previous_hash})"
                )

    is_valid = len(errors) == 0
    return is_valid, errors

"""Audit event serialization and deserialization."""

from __future__ import annotations

import json
from datetime import datetime

from ehr_copilot.domain.audit import AuditEntry, AuditEventType


def serialize_audit_entry(entry: AuditEntry) -> dict:
    """Convert an AuditEntry to a dict suitable for storage."""
    return {
        "entry_id": entry.entry_id,
        "session_id": entry.session_id,
        "patient_id": entry.patient_id,
        "event_type": entry.event_type.value,
        "timestamp": entry.timestamp.isoformat(),
        "data": entry.data,
        "previous_hash": entry.previous_hash,
        "entry_hash": entry.entry_hash,
    }


def deserialize_audit_entry(data: dict) -> AuditEntry:
    """Reconstruct an AuditEntry from a dict."""
    return AuditEntry(
        entry_id=data["entry_id"],
        session_id=data["session_id"],
        patient_id=data["patient_id"],
        event_type=AuditEventType(data["event_type"]),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        data=data["data"] if isinstance(data["data"], dict) else json.loads(data["data"]),
        previous_hash=data["previous_hash"],
        entry_hash=data["entry_hash"],
    )


def entry_to_hashable_string(entry: AuditEntry) -> str:
    """Create a deterministic string representation for hashing.

    Produces a JSON string with sorted keys, excluding the entry_hash
    field so the hash can be computed over the remaining content.
    """
    hashable = {
        "entry_id": entry.entry_id,
        "session_id": entry.session_id,
        "patient_id": entry.patient_id,
        "event_type": entry.event_type.value,
        "timestamp": entry.timestamp.isoformat(),
        "data": entry.data,
        "previous_hash": entry.previous_hash,
    }
    return json.dumps(hashable, sort_keys=True, separators=(",", ":"))

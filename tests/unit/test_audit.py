"""Unit tests for the audit subsystem."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio

from ehr_copilot.audit.integrity import compute_entry_hash, verify_hash_chain
from ehr_copilot.audit.logger import AuditLogger
from ehr_copilot.audit.schemas import (
    deserialize_audit_entry,
    entry_to_hashable_string,
    serialize_audit_entry,
)
from ehr_copilot.domain.audit import AuditEntry, AuditEventType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def logger(tmp_path: Path) -> AuditLogger:
    """AuditLogger backed by a temporary SQLite database file.

    Using a file-based database instead of :memory: because each
    aiosqlite.connect(\":memory:\") call creates a separate in-memory DB.
    """
    db_path = str(tmp_path / "test_audit.db")
    audit_logger = AuditLogger(db_path=db_path)
    await audit_logger.initialize()
    return audit_logger


def _make_entry(
    entry_id: str = "e1",
    session_id: str = "sess-1",
    patient_id: str = "p1",
    event_type: AuditEventType = AuditEventType.QUERY_RECEIVED,
    data: dict | None = None,
    previous_hash: str = "",
) -> AuditEntry:
    """Helper to create an AuditEntry with a computed hash."""
    entry = AuditEntry(
        entry_id=entry_id,
        session_id=session_id,
        patient_id=patient_id,
        event_type=event_type,
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        data=data or {},
        previous_hash=previous_hash,
        entry_hash="",
    )
    hashable_str = entry_to_hashable_string(entry)
    entry.entry_hash = compute_entry_hash(previous_hash, hashable_str)
    return entry


# ---------------------------------------------------------------------------
# AuditLogger.log
# ---------------------------------------------------------------------------


class TestAuditLoggerLog:
    @pytest.mark.asyncio
    async def test_log_creates_entry_with_hash(self, logger):
        entry = await logger.log(
            session_id="sess-1",
            patient_id="p1",
            event_type=AuditEventType.QUERY_RECEIVED,
            data={"query_id": "q1", "text": "What is the A1c?"},
        )
        assert entry.entry_id  # non-empty
        assert entry.entry_hash  # non-empty
        assert len(entry.entry_hash) == 64  # SHA-256 hex
        assert entry.session_id == "sess-1"
        assert entry.patient_id == "p1"
        assert entry.event_type == AuditEventType.QUERY_RECEIVED
        assert entry.data["query_id"] == "q1"

    @pytest.mark.asyncio
    async def test_log_chain_builds_correctly(self, logger):
        """Multiple log entries should form a valid hash chain."""
        e1 = await logger.log("sess-1", "p1", AuditEventType.QUERY_RECEIVED, {"q": 1})
        e2 = await logger.log("sess-1", "p1", AuditEventType.ROUTE_CLASSIFIED, {"type": "FACTUAL"})
        e3 = await logger.log("sess-1", "p1", AuditEventType.ANSWER_RETURNED, {"a": 1})

        # e1 should have empty previous_hash (first entry)
        assert e1.previous_hash == ""
        # e2's previous_hash should be e1's entry_hash
        assert e2.previous_hash == e1.entry_hash
        # e3's previous_hash should be e2's entry_hash
        assert e3.previous_hash == e2.entry_hash


# ---------------------------------------------------------------------------
# AuditLogger.get_session_entries
# ---------------------------------------------------------------------------


class TestGetSessionEntries:
    @pytest.mark.asyncio
    async def test_get_session_entries(self, logger):
        await logger.log("sess-A", "p1", AuditEventType.PATIENT_LOADED, {})
        await logger.log("sess-B", "p2", AuditEventType.PATIENT_LOADED, {})
        await logger.log("sess-A", "p1", AuditEventType.QUERY_RECEIVED, {"q": "test"})

        entries = await logger.get_session_entries("sess-A")
        assert len(entries) == 2
        assert all(e.session_id == "sess-A" for e in entries)

    @pytest.mark.asyncio
    async def test_empty_session(self, logger):
        entries = await logger.get_session_entries("nonexistent")
        assert entries == []


# ---------------------------------------------------------------------------
# Hash chain integrity verification
# ---------------------------------------------------------------------------


class TestHashChainVerification:
    @pytest.mark.asyncio
    async def test_verify_chain_valid(self, logger):
        await logger.log("sess-1", "p1", AuditEventType.QUERY_RECEIVED, {"q": 1})
        await logger.log("sess-1", "p1", AuditEventType.ROUTE_CLASSIFIED, {})
        await logger.log("sess-1", "p1", AuditEventType.ANSWER_RETURNED, {})

        is_valid = await logger.verify_chain("sess-1")
        assert is_valid is True

    def test_verify_hash_chain_direct(self):
        """Test verify_hash_chain with manually constructed entries."""
        e1 = _make_entry(entry_id="e1", previous_hash="")
        e2 = _make_entry(
            entry_id="e2",
            event_type=AuditEventType.ROUTE_CLASSIFIED,
            previous_hash=e1.entry_hash,
        )
        is_valid, errors = verify_hash_chain([e1, e2])
        assert is_valid is True
        assert errors == []

    def test_verify_hash_chain_tampered(self):
        """Tampering with an entry should break the chain."""
        e1 = _make_entry(entry_id="e1", previous_hash="")
        e2 = _make_entry(
            entry_id="e2",
            event_type=AuditEventType.ROUTE_CLASSIFIED,
            previous_hash=e1.entry_hash,
        )
        # Tamper with e1's data after hashing
        e1.data = {"TAMPERED": True}
        is_valid, errors = verify_hash_chain([e1, e2])
        assert is_valid is False
        assert len(errors) >= 1


# ---------------------------------------------------------------------------
# Serialization / Deserialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self):
        entry = _make_entry(
            entry_id="e1",
            data={"key": "value", "number": 42},
        )
        serialized = serialize_audit_entry(entry)
        deserialized = deserialize_audit_entry(serialized)

        assert deserialized.entry_id == entry.entry_id
        assert deserialized.session_id == entry.session_id
        assert deserialized.patient_id == entry.patient_id
        assert deserialized.event_type == entry.event_type
        assert deserialized.data == entry.data
        assert deserialized.previous_hash == entry.previous_hash
        assert deserialized.entry_hash == entry.entry_hash

    def test_deserialize_with_json_string_data(self):
        """When data comes from SQLite it may be a JSON string."""
        entry = _make_entry(data={"query": "test"})
        serialized = serialize_audit_entry(entry)
        serialized["data"] = json.dumps(serialized["data"])
        deserialized = deserialize_audit_entry(serialized)
        assert deserialized.data == {"query": "test"}


# ---------------------------------------------------------------------------
# entry_to_hashable_string determinism
# ---------------------------------------------------------------------------


class TestHashableString:
    def test_deterministic(self):
        entry = _make_entry(
            entry_id="e1",
            data={"b": 2, "a": 1},  # Keys out of order
        )
        s1 = entry_to_hashable_string(entry)
        s2 = entry_to_hashable_string(entry)
        assert s1 == s2

    def test_sorted_keys(self):
        entry = _make_entry(data={"zebra": 1, "alpha": 2})
        s = entry_to_hashable_string(entry)
        # JSON with sort_keys=True should put "alpha" before "zebra"
        parsed = json.loads(s)
        data_keys = list(parsed["data"].keys())
        assert data_keys == sorted(data_keys)

    def test_excludes_entry_hash(self):
        entry = _make_entry(data={"x": 1})
        s = entry_to_hashable_string(entry)
        parsed = json.loads(s)
        assert "entry_hash" not in parsed

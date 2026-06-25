"""Append-only audit logger backed by SQLite with hash chain integrity."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import uuid4

import aiosqlite

from ehr_copilot.domain.audit import AuditEntry, AuditEventType, ProvenanceRecord
from ehr_copilot.audit.integrity import compute_entry_hash
from ehr_copilot.audit.schemas import (
    deserialize_audit_entry,
    entry_to_hashable_string,
    serialize_audit_entry,
)


class AuditLogger:
    """Append-only audit logger with hash chain verification.

    Stores audit entries in a SQLite database and maintains a hash chain
    so that tampering with historical entries can be detected.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    async def initialize(self) -> None:
        """Create the audit_entries table if it does not exist."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_entries (
                    entry_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    patient_id TEXT,
                    event_type TEXT,
                    timestamp TEXT,
                    data TEXT,
                    previous_hash TEXT,
                    entry_hash TEXT
                )
                """
            )
            await db.commit()

    async def log(
        self,
        session_id: str,
        patient_id: str,
        event_type: AuditEventType,
        data: dict,
    ) -> AuditEntry:
        """Append a new audit entry to the log.

        Retrieves the last entry's hash to form the chain link, computes
        a SHA-256 hash over the concatenation of the previous hash and the
        deterministic serialization of this entry, then inserts the row.

        Args:
            session_id: The session this event belongs to.
            patient_id: The patient this event relates to.
            event_type: The type of audit event.
            data: Arbitrary event payload.

        Returns:
            The newly created AuditEntry with its computed hash.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # Get the last entry's hash to chain from
            cursor = await db.execute(
                "SELECT entry_hash FROM audit_entries ORDER BY rowid DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            previous_hash = row[0] if row else ""

            # Build the entry (without hash yet so we can serialize for hashing)
            entry = AuditEntry(
                entry_id=str(uuid4()),
                session_id=session_id,
                patient_id=patient_id,
                event_type=event_type,
                timestamp=datetime.utcnow(),
                data=data,
                previous_hash=previous_hash,
                entry_hash="",
            )

            # Compute the hash
            entry_data = entry_to_hashable_string(entry)
            entry.entry_hash = compute_entry_hash(previous_hash, entry_data)

            # Persist
            serialized = serialize_audit_entry(entry)
            await db.execute(
                """
                INSERT INTO audit_entries
                    (entry_id, session_id, patient_id, event_type,
                     timestamp, data, previous_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    serialized["entry_id"],
                    serialized["session_id"],
                    serialized["patient_id"],
                    serialized["event_type"],
                    serialized["timestamp"],
                    json.dumps(serialized["data"]),
                    serialized["previous_hash"],
                    serialized["entry_hash"],
                ),
            )
            await db.commit()

        return entry

    async def get_session_entries(self, session_id: str) -> list[AuditEntry]:
        """Retrieve all audit entries for a given session, ordered by insertion."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM audit_entries WHERE session_id = ? ORDER BY rowid",
                (session_id,),
            )
            rows = await cursor.fetchall()

        return [
            deserialize_audit_entry(dict(row))
            for row in rows
        ]

    async def get_patient_entries(self, patient_id: str) -> list[AuditEntry]:
        """Retrieve all audit entries for a given patient, ordered by insertion."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM audit_entries WHERE patient_id = ? ORDER BY rowid",
                (patient_id,),
            )
            rows = await cursor.fetchall()

        return [
            deserialize_audit_entry(dict(row))
            for row in rows
        ]

    async def get_provenance(self, answer_id: str, session_id: str) -> ProvenanceRecord:
        """Build a provenance trail linking an answer back through its session events.

        Retrieves all session entries up to and including the
        ANSWER_RETURNED event whose data contains the given answer_id,
        then packages them into a ProvenanceRecord.

        Args:
            answer_id: The identifier of the answer to trace.
            session_id: The session in which the answer was produced.

        Returns:
            A ProvenanceRecord containing the ordered chain of events.
        """
        entries = await self.get_session_entries(session_id)

        # Collect entries up to (and including) the matching ANSWER_RETURNED event
        provenance_entries: list[AuditEntry] = []
        query_id = ""
        patient_id = ""

        for entry in entries:
            provenance_entries.append(entry)

            if not patient_id and entry.patient_id:
                patient_id = entry.patient_id

            if entry.event_type == AuditEventType.QUERY_RECEIVED:
                query_id = entry.data.get("query_id", query_id)

            if (
                entry.event_type == AuditEventType.ANSWER_RETURNED
                and entry.data.get("answer_id") == answer_id
            ):
                break

        return ProvenanceRecord(
            answer_id=answer_id,
            query_id=query_id,
            session_id=session_id,
            patient_id=patient_id,
            events=provenance_entries,
        )

    async def verify_chain(self, session_id: str) -> bool:
        """Verify hash chain integrity for all entries in a session.

        Re-fetches entries for the session and checks that every hash
        is consistent with the previous entry's hash and the entry content.

        Returns:
            True if the chain is valid, False otherwise.
        """
        from ehr_copilot.audit.integrity import verify_hash_chain

        entries = await self.get_session_entries(session_id)
        is_valid, _errors = verify_hash_chain(entries)
        return is_valid

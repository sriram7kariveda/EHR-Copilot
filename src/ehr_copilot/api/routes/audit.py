"""Audit trail endpoints -- session history and integrity verification."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from ehr_copilot.api.schemas import AuditResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/{session_id}", response_model=AuditResponse)
async def get_session_audit(session_id: str, request: Request) -> AuditResponse:
    """Retrieve the full audit trail for a session.

    Returns all audit entries associated with the given ``session_id``, along
    with a flag indicating whether the hash chain is intact.
    """
    audit_logger = request.app.state.audit_logger

    try:
        entries = await audit_logger.get_session_entries(session_id)
    except Exception as exc:
        logger.error("Failed to retrieve audit entries: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {exc}")

    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No audit entries found for session '{session_id}'",
        )

    # Verify chain integrity alongside the read.
    try:
        chain_valid = await audit_logger.verify_chain(session_id)
    except Exception:
        logger.warning("Hash chain verification failed", exc_info=True)
        chain_valid = False

    return AuditResponse(
        session_id=session_id,
        entries=[entry.model_dump() for entry in entries],
        chain_valid=chain_valid,
    )


@router.get("/{session_id}/verify")
async def verify_session_chain(session_id: str, request: Request) -> dict:
    """Verify the hash chain integrity for a session's audit trail.

    Returns a JSON object with ``session_id``, ``chain_valid`` (bool), and
    ``entry_count`` so callers can check tamper evidence without fetching
    all the raw entries.
    """
    audit_logger = request.app.state.audit_logger

    try:
        entries = await audit_logger.get_session_entries(session_id)
    except Exception as exc:
        logger.error("Failed to retrieve audit entries: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audit retrieval failed: {exc}")

    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"No audit entries found for session '{session_id}'",
        )

    try:
        chain_valid = await audit_logger.verify_chain(session_id)
    except Exception:
        logger.warning("Hash chain verification failed", exc_info=True)
        chain_valid = False

    return {
        "session_id": session_id,
        "chain_valid": chain_valid,
        "entry_count": len(entries),
    }

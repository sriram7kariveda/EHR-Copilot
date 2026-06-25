"""Health-check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from ehr_copilot.api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health including LLM availability and loaded patients."""

    llm_client = request.app.state.llm_client
    index_registry = request.app.state.index_registry
    settings = request.app.state.settings

    # Probe LLM availability (non-blocking best-effort).
    llm_available = False
    try:
        llm_available = await llm_client.is_available()
    except Exception:
        logger.warning("LLM availability check failed", exc_info=True)

    loaded_patients = index_registry.list_patients()

    return HealthResponse(
        status="ok",
        llm_available=llm_available,
        loaded_patients=loaded_patients,
        version=settings.app.version,
    )


@router.get("/cost")
async def cost_summary() -> dict:
    """Return cumulative API cost tracking summary."""
    from ehr_copilot.llm.anthropic_client import get_cost_tracker
    return get_cost_tracker().summary()

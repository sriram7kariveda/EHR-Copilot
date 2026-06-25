"""FastAPI application factory for the EHR Copilot service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from ehr_copilot.agents.critic import CriticAgent
from ehr_copilot.agents.numeric_validator import NumericValidatorAgent
from ehr_copilot.agents.pipeline import CopilotPipeline
from ehr_copilot.agents.reasoning import ReasoningAgent
from ehr_copilot.agents.retrieval import RetrievalAgent
from ehr_copilot.agents.router import RouterAgent
from ehr_copilot.agents.temporal_validator import TemporalValidatorAgent
from ehr_copilot.api.middleware import (
    PatientScopeMiddleware,
    RequestIDMiddleware,
    TimingMiddleware,
)
from ehr_copilot.api.routes import audit, health, patient, query
from ehr_copilot.audit.logger import AuditLogger
from ehr_copilot.config import Settings, get_settings
from ehr_copilot.indexing.embedding import EmbeddingModel
from ehr_copilot.indexing.hybrid_retriever import HybridRetriever
from ehr_copilot.indexing.index_registry import IndexRegistry
from ehr_copilot.llm import create_llm_client
from ehr_copilot.llm.prompt_engine import PromptEngine

logger = logging.getLogger(__name__)


def _configure_logging(settings: Settings) -> None:
    """Apply the logging configuration from settings."""
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format=settings.logging.format,
        force=True,
    )


def _build_pipeline(
    settings: Settings,
    llm_client,
    prompt_engine: PromptEngine,
    audit_logger: AuditLogger | None = None,
) -> CopilotPipeline:
    """Wire all agents into a CopilotPipeline using the shared LLM client."""

    # A dummy retriever is needed for the RetrievalAgent constructor. The
    # pipeline swaps it out at runtime with the per-patient retriever.  We
    # create a placeholder that will never be called directly.
    retrieval_agent = RetrievalAgent(retriever=None, top_k=settings.indexing.retrieval.final_top_k)  # type: ignore[arg-type]

    router_agent = RouterAgent(llm_client=llm_client, prompt_engine=prompt_engine)
    reasoning_agent = ReasoningAgent(llm_client=llm_client, prompt_engine=prompt_engine)
    temporal_validator = TemporalValidatorAgent(llm_client=llm_client, prompt_engine=prompt_engine)
    numeric_validator = NumericValidatorAgent(llm_client=llm_client, prompt_engine=prompt_engine)
    critic_agent = CriticAgent(llm_client=llm_client, prompt_engine=prompt_engine)

    return CopilotPipeline(
        router=router_agent,
        retrieval=retrieval_agent,
        reasoning=reasoning_agent,
        temporal_validator=temporal_validator,
        numeric_validator=numeric_validator,
        critic=critic_agent,
        config=settings.agents,
        audit_logger=audit_logger,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Parameters
    ----------
    settings:
        Optional pre-built settings instance.  When ``None`` the default
        settings are loaded via :func:`get_settings`.

    Returns
    -------
    FastAPI
        A fully configured application instance ready to be served.
    """
    if settings is None:
        settings = get_settings()

    _configure_logging(settings)

    # ------------------------------------------------------------------
    # Lifespan -- initialise and tear down shared resources
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting EHR Copilot v%s", settings.app.version)

        # -- LLM client --
        llm_client = create_llm_client(
            provider=settings.llm.provider,
            config=settings.llm,
        )
        app.state.llm_client = llm_client

        # -- Embedding model (lazy-loaded on first use) --
        embedding_model = EmbeddingModel(settings.embedding)
        app.state.embedding_model = embedding_model

        # -- Index registry --
        index_registry = IndexRegistry()
        app.state.index_registry = index_registry

        # -- Audit logger --
        audit_db_path = Path(settings.audit.db_path)
        audit_db_path.parent.mkdir(parents=True, exist_ok=True)
        audit_logger = AuditLogger(db_path=str(audit_db_path))
        await audit_logger.initialize()
        app.state.audit_logger = audit_logger

        # -- Prompt engine --
        prompt_engine = PromptEngine()
        app.state.prompt_engine = prompt_engine

        # -- Agent pipeline --
        pipeline = _build_pipeline(settings, llm_client, prompt_engine, audit_logger)
        app.state.pipeline = pipeline

        # -- Patient context cache (patient_id -> PatientContext) --
        app.state.patient_contexts = {}

        # -- Settings reference --
        app.state.settings = settings

        logger.info("EHR Copilot initialised successfully")

        yield

        # -- Cleanup --
        logger.info("Shutting down EHR Copilot")
        index_registry.destroy_all()
        logger.info("All patient indices destroyed")

    # ------------------------------------------------------------------
    # Build the FastAPI app
    # ------------------------------------------------------------------

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Clinical EHR Copilot -- retrieval-augmented clinical Q&A with citations and audit",
        lifespan=lifespan,
    )

    # -- Middleware (order matters: outermost first) --
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(PatientScopeMiddleware)

    # -- Routes --
    app.include_router(health.router)
    app.include_router(patient.router)
    app.include_router(query.router)
    app.include_router(audit.router)

    return app

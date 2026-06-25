"""Agent base classes with typed generics for the multi-agent pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentContext(BaseModel):
    """Shared context passed through the pipeline."""

    session_id: str
    patient_id: str
    query_id: str

    model_config = {"arbitrary_types_allowed": True}


class AgentResult(BaseModel, Generic[OutputT]):
    """Typed wrapper around an agent's output, carrying metadata."""

    agent_name: str
    output: OutputT
    latency_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)


class AgentBase(ABC, Generic[InputT, OutputT]):
    """Abstract base for every agent in the pipeline.

    Subclasses declare their *InputT* and *OutputT* type parameters and
    implement the asynchronous :meth:`run` method.
    """

    name: str = "base_agent"

    @abstractmethod
    async def run(
        self,
        input_data: InputT,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        ...

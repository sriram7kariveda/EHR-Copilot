from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class LLMRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    stop_sequences: list[str] = Field(default_factory=list)


class LLMResponse(BaseModel):
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class LLMClient(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        ...

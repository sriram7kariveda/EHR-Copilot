"""Ollama LLM client wrapping the official ``ollama`` async Python client."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from ollama import AsyncClient

from ehr_copilot.config import LLMConfig
from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OllamaClient(LLMClient):
    """Async wrapper around a locally-running Ollama instance."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = AsyncClient(
            host=config.base_url,
            timeout=httpx.Timeout(timeout=float(config.timeout_seconds)),
        )

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Send a prompt to Ollama and return the generated text.

        The method uses ``ollama.chat`` so that a *system* message can be
        supplied naturally alongside the user message.  If the caller did
        not provide a system prompt the system message is simply omitted.
        """
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        }
        if request.stop_sequences:
            options["stop"] = request.stop_sequences

        model = self.config.model

        start = time.perf_counter()
        try:
            response = await self._client.chat(
                model=model,
                messages=messages,
                options=options,
            )
        except httpx.ConnectError as exc:
            logger.error("Ollama connection failed at %s: %s", self.config.base_url, exc)
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is the Ollama server running?"
            ) from exc
        except httpx.TimeoutException as exc:
            logger.error(
                "Ollama request timed out after %ss: %s",
                self.config.timeout_seconds,
                exc,
            )
            raise TimeoutError(
                f"Ollama request timed out after {self.config.timeout_seconds}s"
            ) from exc
        except Exception as exc:
            logger.error("Unexpected Ollama error: %s", exc)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000

        # The ollama library returns a dict-like ChatResponse.
        message_content: str = response["message"]["content"]
        prompt_tokens: int = response.get("prompt_eval_count", 0) or 0
        completion_tokens: int = response.get("eval_count", 0) or 0

        return LLMResponse(
            text=message_content,
            model=response.get("model", model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(elapsed_ms, 2),
        )

    async def is_available(self) -> bool:
        """Check whether the Ollama server is reachable by listing models."""
        try:
            await self._client.list()
            return True
        except Exception:  # noqa: BLE001
            logger.debug(
                "Ollama health-check failed at %s", self.config.base_url
            )
            return False

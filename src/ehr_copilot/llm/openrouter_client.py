"""OpenRouter API client - calls Qwen and other models via OpenRouter's OpenAI-compatible API."""

from __future__ import annotations

import logging
import time

import httpx

from ehr_copilot.config import LLMConfig
from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterClient(LLMClient):
    """LLM client that calls models via the OpenRouter API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://ehr-copilot.local",
                    "X-Title": "EHR Copilot",
                },
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from OpenRouter."""
        client = self._get_client()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        start = time.monotonic()
        try:
            resp = await client.post("/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("OpenRouter API error: %s %s", e.response.status_code, e.response.text)
            raise
        except httpx.RequestError as e:
            logger.error("OpenRouter connection error: %s", e)
            raise

        latency_ms = (time.monotonic() - start) * 1000

        choice = data["choices"][0]
        usage = data.get("usage", {})

        # Some models (e.g. MiniMax M2.5) return output in reasoning field
        msg = choice["message"]
        text = msg.get("content") or msg.get("reasoning") or ""

        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
        )

    async def is_available(self) -> bool:
        """Check if OpenRouter API is reachable."""
        try:
            client = self._get_client()
            resp = await client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

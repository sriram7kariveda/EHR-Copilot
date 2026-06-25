"""vLLM LLM Client — connects to a locally served model via OpenAI-compatible API.

Used for MAD debate agents running Qwen 3.5 4B on HPC A100 via vLLM.
"""

from __future__ import annotations

import logging
import time

import httpx

from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class VLLMClient(LLMClient):
    """LLM client that calls a local vLLM server (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "Qwen/Qwen3.5-4B",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(self, request: LLMRequest) -> LLMResponse:
        start = time.perf_counter()

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop_sequences:
            payload["stop"] = request.stop_sequences

        try:
            resp = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            elapsed = (time.perf_counter() - start) * 1000

            return LLMResponse(
                text=text,
                model=data.get("model", self._model),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_ms=round(elapsed, 2),
            )

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error("vLLM request failed: %s", exc)
            return LLMResponse(
                text=f"Error: {exc}",
                model=self._model,
                latency_ms=round(elapsed, 2),
            )

    async def is_available(self) -> bool:
        try:
            resp = await self._client.get(f"{self._base_url}/models")
            return resp.status_code == 200
        except Exception:
            return False

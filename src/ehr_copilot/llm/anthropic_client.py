"""Anthropic API client - calls Claude models directly via the Messages API."""

from __future__ import annotations

import logging
import threading
import time

import httpx

from ehr_copilot.config import LLMConfig
from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# Pricing per million tokens (Haiku 4.5)
_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
}
_DEFAULT_PRICING = {"input": 0.80, "output": 4.00}  # Haiku fallback


class BudgetExceededError(Exception):
    """Raised when the spending cap has been reached."""


class CostTracker:
    """Thread-safe cumulative cost tracker for API calls."""

    def __init__(self, budget_usd: float = 5.0) -> None:
        self.budget_usd = budget_usd
        self._lock = threading.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.total_calls = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return the cost of this call. Raises if over budget."""
        pricing = _PRICING.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
            self.total_calls += 1

            if self.total_cost_usd > self.budget_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self.total_cost_usd:.4f} / ${self.budget_usd:.2f} "
                    f"after {self.total_calls} calls "
                    f"({self.total_input_tokens} in, {self.total_output_tokens} out)"
                )

            if self.total_calls % 10 == 0:
                logger.info(
                    "Cost tracker: $%.4f / $%.2f (%d calls, %d in, %d out)",
                    self.total_cost_usd, self.budget_usd,
                    self.total_calls, self.total_input_tokens, self.total_output_tokens,
                )

        return cost

    def summary(self) -> dict:
        with self._lock:
            return {
                "total_cost_usd": round(self.total_cost_usd, 4),
                "budget_usd": self.budget_usd,
                "budget_remaining_usd": round(self.budget_usd - self.total_cost_usd, 4),
                "total_calls": self.total_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            }


# Global cost tracker shared across all Anthropic client instances
_global_cost_tracker = CostTracker(budget_usd=5.0)


def get_cost_tracker() -> CostTracker:
    """Return the global cost tracker."""
    return _global_cost_tracker


class AnthropicClient(LLMClient):
    """LLM client that calls Claude models via the Anthropic Messages API."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.api_key = config.api_key
        self.model = config.model
        self._client: httpx.AsyncClient | None = None
        self._cost_tracker = _global_cost_tracker

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the Anthropic API."""
        # Pre-flight budget check
        remaining = self._cost_tracker.budget_usd - self._cost_tracker.total_cost_usd
        if remaining <= 0:
            raise BudgetExceededError(
                f"Budget exhausted: ${self._cost_tracker.total_cost_usd:.4f} spent"
            )

        client = self._get_client()

        payload: dict = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt
        if request.stop_sequences:
            payload["stop_sequences"] = request.stop_sequences

        start = time.monotonic()
        try:
            resp = await client.post(ANTHROPIC_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("Anthropic API error: %s %s", e.response.status_code, e.response.text)
            raise
        except httpx.RequestError as e:
            logger.error("Anthropic connection error: %s", e)
            raise

        latency_ms = (time.monotonic() - start) * 1000

        # Extract text from content blocks
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        # Track cost (raises BudgetExceededError if over limit)
        call_cost = self._cost_tracker.record(self.model, input_tokens, output_tokens)
        logger.debug(
            "API call: %d in + %d out = $%.4f (total: $%.4f)",
            input_tokens, output_tokens, call_cost, self._cost_tracker.total_cost_usd,
        )

        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    async def is_available(self) -> bool:
        """Check if Anthropic API is reachable with a minimal request."""
        try:
            client = self._get_client()
            resp = await client.post(
                ANTHROPIC_API_URL,
                json={
                    "model": self.model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

"""Mock LLM client for testing without a real LLM backend."""

from __future__ import annotations

from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse


class MockLLMClient(LLMClient):
    """Mock LLM that returns configurable canned responses."""

    def __init__(self, **kwargs) -> None:
        self.default_response = kwargs.get("default_response", '{"query_type": "FACTUAL"}')
        self.responses: dict[str, str] = kwargs.get("responses", {})
        self.call_log: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_log.append(request)
        # Check if any keyword in responses dict matches the prompt
        for keyword, response_text in self.responses.items():
            if keyword.lower() in request.prompt.lower():
                return LLMResponse(text=response_text, model="mock")
        return LLMResponse(text=self.default_response, model="mock")

    async def is_available(self) -> bool:
        return True

    def reset(self) -> None:
        self.call_log.clear()

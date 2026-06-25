"""LLM abstraction layer with pluggable providers."""

from ehr_copilot.llm.base import LLMClient, LLMRequest, LLMResponse


def create_llm_client(provider: str, **kwargs) -> LLMClient:
    """Factory to create the appropriate LLM client based on provider config."""
    if provider == "openrouter":
        from ehr_copilot.llm.openrouter_client import OpenRouterClient
        from ehr_copilot.config import LLMConfig

        config = kwargs.get("config") or LLMConfig(**kwargs)
        return OpenRouterClient(config)
    elif provider == "ollama":
        from ehr_copilot.llm.ollama_client import OllamaClient
        from ehr_copilot.config import LLMConfig

        config = kwargs.get("config") or LLMConfig(**kwargs)
        return OllamaClient(config)
    elif provider == "anthropic":
        from ehr_copilot.llm.anthropic_client import AnthropicClient
        from ehr_copilot.config import LLMConfig

        config = kwargs.get("config") or LLMConfig(**kwargs)
        return AnthropicClient(config)
    elif provider == "mock":
        from ehr_copilot.llm.mock_client import MockLLMClient

        return MockLLMClient(**kwargs)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


__all__ = ["LLMClient", "LLMRequest", "LLMResponse", "create_llm_client"]

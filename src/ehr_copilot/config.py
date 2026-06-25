"""Pydantic Settings loader with YAML + env overlay."""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


class AppConfig(BaseModel):
    name: str = "EHR Copilot"
    version: str = "0.1.0"
    debug: bool = False


class LLMConfig(BaseModel):
    provider: str = "openrouter"  # "ollama" or "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""  # Required for OpenRouter, set via env EHR__LLM__API_KEY
    model: str = "qwen/qwen3-8b"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout_seconds: int = 300


class EmbeddingConfig(BaseModel):
    model: str = "NeuML/pubmedbert-base-embeddings"
    dimension: int = 768
    batch_size: int = 32
    device: str = "cpu"


class FAISSConfig(BaseModel):
    index_type: str = "FlatIP"
    nprobe: int = 10


class BM25Config(BaseModel):
    k1: float = 1.5
    b: float = 0.75


class RerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enabled: bool = True
    top_k_rerank: int = 50  # Number of RRF candidates to rerank


class RetrievalConfig(BaseModel):
    top_k_dense: int = 30
    top_k_sparse: int = 30
    rrf_k: int = 20
    final_top_k: int = 15
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)


class IndexingConfig(BaseModel):
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)


class ChunkingConfig(BaseModel):
    max_chunk_tokens: int = 512
    overlap_tokens: int = 128
    min_chunk_tokens: int = 50


class AgentTemplateConfig(BaseModel):
    prompt_template: str = ""


class AgentsConfig(BaseModel):
    max_retry_loops: int = 2
    router: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig)
    reasoning: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig)
    temporal_validator: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig)
    numeric_validator: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig)
    critic: AgentTemplateConfig = Field(default_factory=AgentTemplateConfig)


class AuditConfig(BaseModel):
    db_path: str = "./data/audit.db"
    enabled: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


class Settings(BaseSettings):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = {
        "env_prefix": "EHR_",
        "env_nested_delimiter": "__",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


def load_yaml_config(profile: str = "settings") -> dict:
    """Load a YAML config file from the config directory."""
    path = CONFIG_DIR / f"{profile}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base dict, recursively for nested dicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache
def get_settings(profile: str = "settings") -> Settings:
    """Load settings: env vars > .env file > YAML defaults.

    pydantic-settings reads env vars and .env automatically.
    We load YAML as fallback defaults, then let Settings() override with env.
    """
    yaml_data = load_yaml_config(profile)

    # Build settings from env first (pydantic-settings handles env + .env)
    env_settings = Settings()

    # Merge: start from YAML, overlay with any non-default env values
    # Simplest approach: just let env vars win via pydantic-settings
    # Pass YAML as _defaults_ by using model_validate
    env_dict = env_settings.model_dump()
    merged = _deep_merge(yaml_data, env_dict)
    return Settings(**merged)

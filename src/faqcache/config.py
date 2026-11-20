"""All settings in one place (12-factor style: read from environment variables or a .env file).

Every variable starts with FAQ_, for example FAQ_THRESHOLD=0.85 or FAQ_LLM_PROVIDER=anthropic.
API keys are NOT stored here: each provider SDK reads its own variable
(ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DAY_SECONDS = 86_400


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FAQ_", env_file=".env", extra="ignore")

    # --- embeddings -------------------------------------------------------------------------
    embedder: Literal["sentence-transformers", "hash"] = "sentence-transformers"
    # Best safe result in reports/MODEL_COMPARISON.md. Lighter option (470 MB vs 1.1 GB):
    # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 - re-run eval, its best T differs.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    # --- cache ------------------------------------------------------------------------------
    store: Literal["memory", "faiss"] = "memory"
    seed_variants: bool = True  # also cache each FAQ's extra phrasings (data/faqs.json)
    threshold: float = Field(0.83, ge=0.0, le=1.0)  # chosen by `faqcache eval` on the dev half
    top_k: int = Field(3, ge=1, le=20)
    negation_guard: bool = True
    fallthrough: bool = True  # if the best match is blocked by a guard, try the next one
    volatile_ttl_days: float | None = 7.0  # prices / fees / policies expire after N days
    default_ttl_days: float | None = None  # None = never expires
    default_tenant: str = "demomart"

    # --- LLM --------------------------------------------------------------------------------
    llm_provider: Literal["fake", "anthropic", "openai", "gemini", "ollama"] = "fake"
    llm_model: str | None = None  # provider default if empty (Anthropic: claude-opus-5-5)
    llm_effort: Literal["low", "medium", "high"] = "low"  # Anthropic only
    ollama_host: str = "http://localhost:11434"
    fake_llm_latency_ms: float = Field(800.0, ge=0.0)

    # --- files ------------------------------------------------------------------------------
    faqs_path: Path = Path("data/faqs.json")
    test_set_path: Path = Path("data/test_questions.jsonl")
    log_path: Path | None = Path("logs/requests.jsonl")
    reports_dir: Path = Path("reports")

    @field_validator(
        "log_path", "volatile_ttl_days", "default_ttl_days", "llm_model", mode="before"
    )
    @classmethod
    def _empty_means_none(cls, v):
        """FAQ_LOG_PATH= (empty) turns logging off instead of meaning the folder '.'."""
        return None if isinstance(v, str) and v.strip().lower() in ("", "none") else v

    @property
    def volatile_ttl_s(self) -> float | None:
        return None if self.volatile_ttl_days is None else self.volatile_ttl_days * DAY_SECONDS

    @property
    def default_ttl_s(self) -> float | None:
        return None if self.default_ttl_days is None else self.default_ttl_days * DAY_SECONDS


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

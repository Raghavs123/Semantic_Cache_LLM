"""LLM providers behind one tiny interface: generate(question, system) -> LLMResponse."""

from __future__ import annotations

from faqcache.llm.base import LLMClient, LLMResponse
from faqcache.llm.fake import FakeLLM


def build_llm(settings, faqs=None) -> LLMClient:
    provider = settings.llm_provider
    if provider == "fake":
        return FakeLLM(latency_ms=settings.fake_llm_latency_ms, faqs=faqs or [])
    if provider == "anthropic":
        from faqcache.llm.anthropic_client import AnthropicLLM

        return AnthropicLLM(
            model=settings.llm_model or "claude-opus-5-5", effort=settings.llm_effort
        )
    if not settings.llm_model:
        raise ValueError(f"Set FAQ_LLM_MODEL for provider '{provider}' (see .env.example).")
    if provider == "openai":
        from faqcache.llm.openai_client import OpenAILLM

        return OpenAILLM(model=settings.llm_model)
    if provider == "gemini":
        from faqcache.llm.gemini_client import GeminiLLM

        return GeminiLLM(model=settings.llm_model)
    if provider == "ollama":
        from faqcache.llm.ollama_client import OllamaLLM

        return OllamaLLM(model=settings.llm_model, host=settings.ollama_host)
    raise ValueError(f"unknown provider {provider!r}")


__all__ = ["FakeLLM", "LLMClient", "LLMResponse", "build_llm"]

"""Offline stand-in LLM, for tests, CI and demos without an API key.

It is NOT a language model. It sleeps `latency_ms` (to imitate a network call) and returns a
clearly labelled placeholder answer. Any latency number produced with it is a SIMULATION and
must be labelled as such.
"""

from __future__ import annotations

import time

from faqcache.llm.base import LLMResponse


class FakeLLM:
    name = "fake"

    def __init__(self, latency_ms: float = 800.0, faqs=(), cacheable: bool = True) -> None:
        self.latency_ms = latency_ms
        self.faqs = list(faqs)
        self.cacheable = cacheable
        self.calls = 0

    def generate(self, question: str, system: str) -> LLMResponse:
        self.calls += 1
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        text = f"[offline stand-in LLM answer #{self.calls} to: {question}]"
        return LLMResponse(
            text=text,
            input_tokens=(len(system) + len(question)) // 4,  # rough 4-chars-per-token estimate
            output_tokens=len(text) // 4,
            model="fake-offline",
            cacheable=self.cacheable,
        )

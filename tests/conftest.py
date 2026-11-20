"""Shared fixtures. Tests use the offline HashEmbedder and FakeLLM: no download, no API key."""

from __future__ import annotations

from pathlib import Path

import pytest

from faqcache.cache import SemanticCache
from faqcache.data import FAQ
from faqcache.embeddings import HashEmbedder
from faqcache.llm.base import build_system_prompt
from faqcache.llm.fake import FakeLLM
from faqcache.service import FAQBot
from faqcache.stores.memory import InMemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def faqs() -> list[FAQ]:
    return [
        FAQ(id="F01", category="orders", question="How can I track my order?",
            answer="Open My Orders and tap Track."),
        FAQ(id="F02", category="refunds", question="How long does a refund take?",
            answer="Refunds take 5-7 working days."),
        FAQ(id="F03", category="refunds", question="I have not received my refund yet",
            answer="Raise a ticket from the order page."),
        FAQ(id="F11", category="delivery", question="What are the delivery charges?",
            answer="Free above Rs 499, else Rs 40."),
    ]  # fmt: skip


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def cache(embedder, clock, faqs) -> SemanticCache:
    c = SemanticCache(embedder, InMemoryVectorStore(embedder.dim), threshold=0.6, clock=clock)
    c.seed(faqs)
    return c


@pytest.fixture
def bot(embedder, faqs):
    llm = FakeLLM(latency_ms=0, faqs=faqs)

    def factory(tenant: str) -> SemanticCache:
        c = SemanticCache(embedder, InMemoryVectorStore(embedder.dim), threshold=0.6)
        if tenant == "demomart":
            c.seed(faqs)
        return c

    return FAQBot(factory, llm, system_prompt=build_system_prompt(faqs))

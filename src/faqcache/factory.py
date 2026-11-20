"""Composition root: the only place that wires settings -> concrete classes."""

from __future__ import annotations

from faqcache.cache import SemanticCache
from faqcache.config import Settings
from faqcache.data import FAQ, load_faqs
from faqcache.embeddings import Embedder, build_embedder
from faqcache.guards import TTLPolicy
from faqcache.llm import build_llm
from faqcache.llm.base import LLMClient, build_system_prompt
from faqcache.service import FAQBot, RequestLogger
from faqcache.stores import build_store


def build_cache(settings: Settings, embedder: Embedder, tenant: str = "default") -> SemanticCache:
    return SemanticCache(
        embedder,
        build_store(settings.store, embedder.dim),
        threshold=settings.threshold,
        top_k=settings.top_k,
        negation_guard=settings.negation_guard,
        fallthrough=settings.fallthrough,
        ttl=TTLPolicy(settings.volatile_ttl_s, settings.default_ttl_s),
        namespace=tenant,
    )


def build_bot(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
    faqs: list[FAQ] | None = None,
    seed: bool = True,
) -> FAQBot:
    faqs = faqs if faqs is not None else load_faqs(settings.faqs_path)
    embedder = embedder or build_embedder(settings.embedder, settings.embedding_model)
    for text in ("warm-up", "warm up the model", "a slightly longer warm-up sentence here"):
        embedder.embed([text])  # first calls are slow (lazy init); pay that at start-up

    def cache_factory(tenant: str) -> SemanticCache:
        cache = build_cache(settings, embedder, tenant)
        if seed and tenant == settings.default_tenant:
            # only the demo tenant gets the DemoMart FAQs
            cache.seed(faqs, include_variants=settings.seed_variants)
        return cache

    bot = FAQBot(
        cache_factory,
        llm or build_llm(settings, faqs),
        system_prompt=build_system_prompt(faqs),
        logger=RequestLogger(settings.log_path),
        default_tenant=settings.default_tenant,
    )
    bot.cache_for()  # build + seed the default tenant now, so the 1st question is not timed with it
    return bot

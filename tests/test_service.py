from faqcache.cache import SemanticCache
from faqcache.llm.fake import FakeLLM
from faqcache.service import FAQBot
from faqcache.stores.memory import InMemoryVectorStore


def test_seeded_question_hits_cache_without_llm(bot):
    ans = bot.ask("How can I track my order?")
    assert ans.source == "cache" and ans.input_tokens == 0
    assert bot.llm.calls == 0


def test_miss_calls_llm_then_repeat_hits(bot):
    first = bot.ask("Do you deliver on Sundays?")
    assert first.source == "llm" and first.stored and bot.llm.calls == 1
    again = bot.ask("do you deliver on sundays")
    assert again.source == "cache" and bot.llm.calls == 1


def test_miss_embeds_the_question_only_once(bot):
    cache = bot.cache_for()
    calls = []
    real = cache.embedder.embed
    cache.embedder.embed = lambda texts: calls.append(len(texts)) or real(texts)
    try:
        assert bot.ask("Do you sell medicines?").stored
    finally:
        cache.embedder.embed = real
    assert calls == [1]  # lookup embedded it; put() reused that vector


def test_personal_question_bypasses_cache(bot):
    ans = bot.ask("Where is order ORD-123456?")
    assert ans.source == "llm" and ans.reason == "user_data" and not ans.stored


def test_tenants_are_isolated(bot):
    bot.ask("Do you deliver on Sundays?", tenant="shop-a")
    other = bot.ask("Do you deliver on Sundays?", tenant="shop-b")
    assert other.source == "llm"
    assert bot.ask("How can I track my order?", tenant="shop-b").source == "llm"  # no seed


def test_uncacheable_llm_answer_is_not_stored(embedder, faqs):
    llm = FakeLLM(latency_ms=0, faqs=faqs, cacheable=False)  # e.g. a refusal or truncation

    def factory(t):
        return SemanticCache(embedder, InMemoryVectorStore(embedder.dim))

    b = FAQBot(factory, llm, system_prompt="x")
    assert not b.ask("Do you sell medicines?").stored
    assert b.ask("Do you sell medicines?").source == "llm"


def test_stats(bot):
    bot.ask("How can I track my order?")
    bot.ask("Write me a poem about the sea")
    s = bot.logger.stats()
    assert s["requests"] == 2 and s["hits"] == 1 and s["hit_rate"] == 0.5

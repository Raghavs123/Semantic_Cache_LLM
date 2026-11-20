import numpy as np
import pytest

from faqcache.cache import SemanticCache, decide, normalize
from faqcache.stores.base import CacheEntry, Candidate
from faqcache.stores.memory import InMemoryVectorStore


def _cand(q: str, score: float, fid: str = "F") -> Candidate:
    return Candidate(CacheEntry(fid, q, "a", 0.0, None, "faq", fid), score)


def test_normalize():
    assert normalize("  How can I TRACK my order?? ") == "how can i track my order"


def test_exact_hit_is_layer_one(cache):
    res = cache.lookup("how can i track my order")
    assert res.hit and res.reason == "exact_hit" and res.score == 1.0


def test_semantic_hit_and_miss(cache):
    hit = cache.lookup("How can I track my orders?")
    assert hit.hit and hit.entry.faq_id == "F01" and hit.reason == "semantic_hit"
    miss = cache.lookup("What is the capital of Australia?")
    assert not miss.hit and miss.reason == "below_threshold"


def test_vectors_are_unit_length(embedder):
    v = embedder.embed(["hello world", "refund kab milega"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_decide_threshold_boundary_is_inclusive():
    assert decide("q", [_cand("q", 0.80)], 0.80).hit
    assert not decide("q", [_cand("q", 0.7999)], 0.80).hit


def test_negation_guard_blocks_and_fallthrough_rescues():
    cands = [_cand("refund milega", 0.95, "A"), _cand("refund nahi milega", 0.90, "B")]
    q = "refund nahi mila"
    no_guard = decide(q, cands, 0.8, negation_guard=False)
    assert no_guard.hit and no_guard.candidate.entry.faq_id == "A"
    skip = decide(q, cands, 0.8, fallthrough=False)
    assert not skip.hit and skip.reason == "negation_guard"
    fall = decide(q, cands, 0.8, fallthrough=True)
    assert fall.hit and fall.candidate.entry.faq_id == "B" and len(fall.blocked) == 1


def test_seed_with_variants_adds_one_entry_per_phrasing(embedder):
    from faqcache.data import FAQ

    faq = FAQ(
        id="F01",
        category="c",
        question="How can I track my order?",
        answer="Tap Track.",
        variants=["Order ka status kaise dekhe?", "What is the status of my order?"],
    )
    c = SemanticCache(embedder, InMemoryVectorStore(embedder.dim), threshold=0.6)
    assert c.seed([faq], include_variants=False) == 1
    assert c.seed([faq]) == 2  # main question is a duplicate now, the 2 variants are new
    res = c.lookup("order ka status kaise dekhe")
    assert res.hit and res.entry.faq_id == "F01" and res.entry.answer == "Tap Track."


def test_empty_cache(embedder):
    c = SemanticCache(embedder, InMemoryVectorStore(embedder.dim))
    assert c.lookup("anything").reason == "empty_cache"


def test_user_data_is_never_stored(cache):
    res = cache.put("Where is order ORD-555555?", "It is out for delivery")
    assert not res.stored and res.reason == "user_data:order_id"
    res = cache.put("When is my refund coming?", "Hi Rahul, it is on the way")
    assert not res.stored


def test_duplicate_not_stored_twice(cache):
    n = len(cache)
    assert cache.put("How can I track my order?", "x").reason == "duplicate"
    assert len(cache) == n


def test_ttl_expiry_with_fake_clock(cache, clock):
    # delivery charges are 'volatile' -> expires after 7 days (default policy)
    assert cache.lookup("What are the delivery charges?").hit
    clock.t += 8 * 86_400
    res = cache.lookup("What are the delivery charges?")
    assert not res.hit
    # non-volatile entries never expire
    assert cache.lookup("How can I track my order?").hit


def test_evict_expired(cache, clock):
    clock.t += 8 * 86_400
    assert cache.evict_expired() >= 1


def test_faiss_store_matches_memory_store(embedder, faqs):
    pytest.importorskip("faiss")
    from faqcache.stores.faiss_store import FaissVectorStore

    results = []
    for store in (InMemoryVectorStore(embedder.dim), FaissVectorStore(embedder.dim)):
        c = SemanticCache(embedder, store, threshold=0.5, top_k=3)
        c.seed(faqs)
        res = c.lookup("how long do refunds take")
        results.append([(x.entry.faq_id, round(x.score, 4)) for x in res.candidates])
        c.store.remove(res.candidates[0].entry.id)
        assert len(c.store) == len(faqs) - 1
    assert results[0] == results[1]

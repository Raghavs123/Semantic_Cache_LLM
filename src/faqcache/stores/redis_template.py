"""PRODUCTION-STYLE OPTION: Redis + RedisVL SemanticCache.

TEMPLATE - NOT RUN in this repo (no Redis server was available when it was built).
Use it as a starting point; check the RedisVL docs for your installed version, because the
import path moved between releases (older: redisvl.extensions.llmcache).

Start Redis:  docker compose up -d redis      (see docker-compose.yml)
Install:      pip install -e ".[redis]"

Key differences from the in-memory cache:
* RedisVL uses a cosine DISTANCE threshold: distance = 1 - similarity.
  A similarity threshold of 0.80 becomes distance_threshold = 0.20.
* TTL is native (`ttl=` seconds); Redis deletes expired keys for you.
* One index name per tenant keeps customers' caches apart.
"""

from __future__ import annotations

from faqcache.guards import find_user_data, negation_mismatch


def make_redis_cache(tenant: str, similarity_threshold: float, redis_url: str, model: str):
    try:
        from redisvl.extensions.cache.llm import SemanticCache  # newer RedisVL
    except ImportError:  # pragma: no cover
        from redisvl.extensions.llmcache import SemanticCache  # older RedisVL
    from redisvl.utils.vectorize import HFTextVectorizer

    return SemanticCache(
        name=f"faqcache:{tenant}",
        redis_url=redis_url,
        distance_threshold=1.0 - similarity_threshold,
        vectorizer=HFTextVectorizer(model=model),
    )


def redis_lookup(cache, question: str) -> str | None:
    """Return a cached answer or None, applying the same guards as the in-memory cache."""
    if find_user_data(question):
        return None
    for hit in cache.check(prompt=question, num_results=3):
        if not negation_mismatch(question, hit["prompt"]):
            return hit["response"]
    return None


def redis_store(cache, question: str, answer: str, ttl_s: int | None) -> bool:
    if find_user_data(question) or find_user_data(answer):
        return False
    cache.store(prompt=question, response=answer, ttl=ttl_s)
    return True

"""The FAQ bot: cache first, LLM on a miss, log every request."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from faqcache.cache import SemanticCache
from faqcache.guards import find_user_data, redact
from faqcache.llm.base import LLMClient


@dataclass
class Answer:
    answer: str
    source: str  # "cache" or "llm"
    reason: str  # exact_hit | semantic_hit | below_threshold | negation_guard | user_data | ...
    similarity: float | None
    matched_question: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int
    stored: bool
    tenant: str
    lookup_ms: float = 0.0  # time spent in the cache (embed + search + guards)
    llm_ms: float = 0.0  # time spent waiting for the LLM


class RequestLogger:
    """Append one JSON line per request. Personal data is masked before writing."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.counts = {"requests": 0, "hits": 0, "misses": 0, "llm_tokens": 0}
        self._latency_sum = {"cache": 0.0, "llm": 0.0}
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, question: str, ans: Answer, provider: str) -> None:
        record = {
            "ts": round(time.time(), 3),
            "tenant": ans.tenant,
            "question": redact(question),
            "hit": ans.source == "cache",
            "reason": ans.reason,
            "similarity": None if ans.similarity is None else round(ans.similarity, 4),
            "matched_question": ans.matched_question,
            "latency_ms": round(ans.latency_ms, 2),
            "input_tokens": ans.input_tokens,
            "output_tokens": ans.output_tokens,
            "provider": provider,
        }
        with self._lock:
            self.counts["requests"] += 1
            self.counts["hits" if record["hit"] else "misses"] += 1
            self.counts["llm_tokens"] += ans.input_tokens + ans.output_tokens
            self._latency_sum[ans.source] += ans.latency_ms
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self) -> dict:
        with self._lock:
            c = dict(self.counts)
            n = c["requests"]
            c["hit_rate"] = round(c["hits"] / n, 4) if n else 0.0
            c["avg_cache_latency_ms"] = (
                round(self._latency_sum["cache"] / c["hits"], 2) if c["hits"] else None
            )
            c["avg_llm_latency_ms"] = (
                round(self._latency_sum["llm"] / c["misses"], 2) if c["misses"] else None
            )
            return c


class FAQBot:
    """One cache per tenant (customer), so company A never gets company B's answers."""

    def __init__(
        self,
        cache_factory: Callable[[str], SemanticCache],
        llm: LLMClient,
        *,
        system_prompt: str,
        logger: RequestLogger | None = None,
        default_tenant: str = "demomart",
    ) -> None:
        self._cache_factory = cache_factory
        self._caches: dict[str, SemanticCache] = {}
        self._lock = threading.Lock()
        self.llm = llm
        self.system_prompt = system_prompt
        self.logger = logger or RequestLogger(None)
        self.default_tenant = default_tenant

    def cache_for(self, tenant: str | None = None) -> SemanticCache:
        tenant = tenant or self.default_tenant
        with self._lock:
            if tenant not in self._caches:
                self._caches[tenant] = self._cache_factory(tenant)
            return self._caches[tenant]

    def ask(self, question: str, tenant: str | None = None, *, use_cache: bool = True) -> Answer:
        tenant = tenant or self.default_tenant
        start = time.perf_counter()
        cache = self.cache_for(tenant)
        personal = find_user_data(question)
        similarity = matched = vector = None
        lookup_ms = 0.0
        reason = "cache_disabled" if not use_cache else "user_data" if personal else ""

        if use_cache and not personal:
            res = cache.lookup(question)
            lookup_ms = res.latency_ms
            similarity = res.score
            if res.hit and res.entry is not None:
                ans = Answer(
                    answer=res.entry.answer,
                    source="cache",
                    reason=res.reason,
                    similarity=res.score,
                    matched_question=res.entry.question,
                    latency_ms=_ms(start),
                    input_tokens=0,
                    output_tokens=0,
                    stored=False,
                    tenant=tenant,
                    lookup_ms=lookup_ms,
                )
                self.logger.log(question, ans, "cache")
                return ans
            reason = res.reason
            vector = res.vector
            if res.candidates:
                matched = res.candidates[0].entry.question

        llm_start = time.perf_counter()
        resp = self.llm.generate(question, self.system_prompt)
        llm_ms = _ms(llm_start)
        stored = False
        if use_cache and not personal and resp.cacheable:
            stored = cache.put(question, resp.text, source="llm", vector=vector).stored
        ans = Answer(
            answer=resp.text,
            source="llm",
            reason=reason,
            similarity=similarity,
            matched_question=matched,
            latency_ms=_ms(start),
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            stored=stored,
            tenant=tenant,
            lookup_ms=lookup_ms,
            llm_ms=llm_ms,
        )
        self.logger.log(question, ans, self.llm.name)
        return ans


def answer_dict(ans: Answer) -> dict:
    d = asdict(ans)
    for k in ("latency_ms", "lookup_ms", "llm_ms"):
        d[k] = round(d[k], 2)
    if d["similarity"] is not None:
        d["similarity"] = round(d["similarity"], 4)
    return d


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0

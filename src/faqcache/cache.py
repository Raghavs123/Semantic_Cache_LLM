"""The semantic cache: two layers in front of the LLM.

    question
       |
       v
  [L1 exact]  same text after normalising?            -> hit (score 1.0)
       |
  [L2 semantic] embed -> top-k nearest -> threshold + guards -> hit
       |
     miss -> caller asks the LLM, then calls put()

`decide()` is a pure function. The evaluation code calls the same function, so the numbers
in the report come from exactly the logic that serves real traffic.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from faqcache.data import FAQ
from faqcache.embeddings import Embedder
from faqcache.guards import TTLPolicy, find_user_data, negation_mismatch
from faqcache.stores.base import CacheEntry, Candidate, VectorStore


def normalize(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return " ".join(text.split())


@dataclass(frozen=True)
class Decision:
    hit: bool
    candidate: Candidate | None  # the chosen entry on a hit, the best one on a miss
    reason: str  # exact_hit | semantic_hit | below_threshold | negation_guard | empty_cache
    blocked: tuple[Candidate, ...] = ()


def decide(
    question: str,
    candidates: list[Candidate],
    threshold: float,
    *,
    negation_guard: bool = True,
    fallthrough: bool = True,
) -> Decision:
    """Pick a cached answer for `question`, or say why not. Candidates are best-first."""
    if not candidates:
        return Decision(False, None, "empty_cache")
    blocked: list[Candidate] = []
    for cand in candidates:
        if cand.score < threshold:
            break
        if negation_guard and negation_mismatch(question, cand.entry.question):
            blocked.append(cand)
            if not fallthrough:
                break
            continue
        return Decision(True, cand, "semantic_hit", tuple(blocked))
    reason = "negation_guard" if blocked else "below_threshold"
    return Decision(False, candidates[0], reason, tuple(blocked))


@dataclass
class LookupResult:
    hit: bool
    reason: str
    entry: CacheEntry | None
    score: float | None
    latency_ms: float
    candidates: list[Candidate] = field(default_factory=list)
    blocked: tuple[Candidate, ...] = ()
    vector: np.ndarray | None = None  # reuse it in put() on a miss: saves a 2nd embedding


@dataclass
class PutResult:
    stored: bool
    reason: str  # stored | duplicate | user_data:<kind>
    entry: CacheEntry | None = None


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        threshold: float = 0.80,
        top_k: int = 3,
        negation_guard: bool = True,
        fallthrough: bool = True,
        ttl: TTLPolicy | None = None,
        clock: Callable[[], float] = time.time,
        namespace: str = "default",
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.threshold = threshold
        self.top_k = top_k
        self.negation_guard = negation_guard
        self.fallthrough = fallthrough
        self.ttl = ttl or TTLPolicy()
        self.clock = clock
        self.namespace = namespace
        self._exact: dict[str, str] = {}  # normalised question -> entry id
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ read
    def lookup(self, question: str) -> LookupResult:
        start = time.perf_counter()
        with self._lock:
            entry = self._live_exact(question)
            if entry is not None:
                cand = Candidate(entry, 1.0)
                return LookupResult(True, "exact_hit", entry, 1.0, _ms(start), [cand])
            vector = self.embedder.embed([question])[0]
            candidates = self._live(self.store.search(vector, self.top_k))
            d = decide(
                question,
                candidates,
                self.threshold,
                negation_guard=self.negation_guard,
                fallthrough=self.fallthrough,
            )
        best = d.candidate
        return LookupResult(
            hit=d.hit,
            reason=d.reason,
            entry=best.entry if (best and d.hit) else None,
            score=best.score if best else None,
            latency_ms=_ms(start),
            candidates=candidates,
            blocked=d.blocked,
            vector=vector,
        )

    # ------------------------------------------------------------------ write
    def put(
        self,
        question: str,
        answer: str,
        *,
        source: str = "llm",
        faq_id: str | None = None,
        vector: np.ndarray | None = None,
    ) -> PutResult:
        kind = find_user_data(question) or find_user_data(answer)
        if kind:
            return PutResult(False, f"user_data:{kind}")
        if vector is None:
            vector = self.embedder.embed([question])[0]
        return self._put_vector(question, answer, vector, source, faq_id)

    def seed(self, faqs: Iterable[FAQ], *, include_variants: bool = True) -> int:
        """Warm the cache with curated FAQ pairs (one batched embedding call)."""
        pairs = [(q, f) for f in faqs for q in f.phrasings(include_variants)]
        vectors = self.embedder.embed([q for q, _ in pairs])
        stored = 0
        for (question, faq), vec in zip(pairs, vectors, strict=True):
            if self._put_vector(question, faq.answer, vec, "faq", faq.id).stored:
                stored += 1
        return stored

    def evict_expired(self) -> int:
        now = self.clock()
        with self._lock:
            dead = [e.id for e in self.store.entries() if e.expired(now)]
            for entry_id in dead:
                self._remove(entry_id)
        return len(dead)

    def clear(self) -> None:
        with self._lock:
            for e in self.store.entries():
                self._remove(e.id)

    def __len__(self) -> int:
        return len(self.store)

    # ------------------------------------------------------------------ internals
    def _put_vector(self, question, answer, vector, source, faq_id) -> PutResult:
        with self._lock:
            if self._live_exact(question) is not None:
                return PutResult(False, "duplicate")
            now = self.clock()
            ttl_s = self.ttl.ttl_for(question, answer)
            entry = CacheEntry(
                id=uuid.uuid4().hex[:12],
                question=question,
                answer=answer,
                created_at=now,
                expires_at=None if ttl_s is None else now + ttl_s,
                source=source,  # type: ignore[arg-type]
                faq_id=faq_id,
            )
            self.store.add(entry, vector)
            self._exact[normalize(question)] = entry.id
            return PutResult(True, "stored", entry)

    def _live_exact(self, question: str) -> CacheEntry | None:
        entry_id = self._exact.get(normalize(question))
        if entry_id is None:
            return None
        entry = self.store.get(entry_id)
        if entry is None:
            self._exact.pop(normalize(question), None)
            return None
        if entry.expired(self.clock()):
            self._remove(entry_id)
            return None
        return entry

    def _live(self, candidates: list[Candidate]) -> list[Candidate]:
        now = self.clock()
        alive = []
        for c in candidates:
            if c.entry.expired(now):
                self._remove(c.entry.id)  # lazy eviction
            else:
                alive.append(c)
        return alive

    def _remove(self, entry_id: str) -> None:
        entry = self.store.get(entry_id)
        if entry is not None:
            self._exact.pop(normalize(entry.question), None)
        self.store.remove(entry_id)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0

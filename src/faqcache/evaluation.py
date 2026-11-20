"""Measure the cache: threshold sweep, dev/test hold-out, guard ablation, end-to-end replay.

Definitions (used everywhere in the reports):
* hit         - the cache returned a saved answer.
* correct hit - a hit whose saved FAQ is the labelled one.
* false hit   - a hit on a question labelled "none", or on the WRONG FAQ.  (the dangerous one)
* hit rate    - hits / all questions. With 60 paraphrases and 40 "none" questions the best
                possible hit rate is 60%.
* recall      - correct hits / paraphrase questions (out of 60).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from faqcache.cache import SemanticCache, decide
from faqcache.data import NONE_LABEL, TestItem
from faqcache.stores.base import Candidate


@dataclass
class Probe:
    """One test question looked up once against the warm cache (top-k kept)."""

    item: TestItem
    candidates: list[Candidate]
    lookup_ms: float  # real measured time: normalise + embed + vector search + guards


@dataclass
class SweepRow:
    threshold: float
    n: int
    hits: int
    hit_rate: float
    correct_hits: int
    false_hits: int
    recall: float
    precision: float
    llm_calls: int
    llm_calls_saved_pct: float
    est_avg_latency_ms: float


def probe_all(cache: SemanticCache, items: list[TestItem], warmup: int = 3) -> list[Probe]:
    for item in items[:warmup]:  # first calls are slower (lazy init); don't time them
        cache.lookup(item.question)
    probes = []
    for item in items:
        res = cache.lookup(item.question)
        probes.append(Probe(item, res.candidates, res.latency_ms))
    return probes


def evaluate(
    probes: list[Probe],
    threshold: float,
    *,
    negation_guard: bool = True,
    fallthrough: bool = True,
    llm_latency_ms: float = 800.0,
) -> tuple[SweepRow, list[dict]]:
    rows, hits, correct, false = [], 0, 0, 0
    positives = sum(p.item.label != NONE_LABEL for p in probes)
    latency = []
    for p in probes:
        d = decide(
            p.item.question,
            p.candidates,
            threshold,
            negation_guard=negation_guard,
            fallthrough=fallthrough,
        )
        matched = d.candidate.entry.faq_id if (d.hit and d.candidate) else None
        is_correct = d.hit and matched == p.item.label
        is_false = d.hit and not is_correct
        hits += d.hit
        correct += is_correct
        false += is_false
        latency.append(p.lookup_ms + (0.0 if d.hit else llm_latency_ms))
        rows.append(
            {
                "id": p.item.id,
                "question": p.item.question,
                "label": p.item.label,
                "kind": p.item.kind,
                "hit": d.hit,
                "reason": d.reason,
                "matched": matched,
                "score": round(d.candidate.score, 4) if d.candidate else None,
                "outcome": "correct_hit"
                if is_correct
                else "FALSE_HIT"
                if is_false
                else "missed"
                if p.item.label != NONE_LABEL
                else "correct_miss",
            }
        )
    n = len(probes)
    row = SweepRow(
        threshold=round(threshold, 4),
        n=n,
        hits=hits,
        hit_rate=round(hits / n, 4) if n else 0.0,
        correct_hits=correct,
        false_hits=false,
        recall=round(correct / positives, 4) if positives else 0.0,
        precision=round(correct / hits, 4) if hits else 1.0,
        llm_calls=n - hits,
        llm_calls_saved_pct=round(100 * hits / n, 2) if n else 0.0,
        est_avg_latency_ms=round(statistics.fmean(latency), 2) if latency else 0.0,
    )
    return row, rows


def sweep(probes, thresholds, **kw) -> list[SweepRow]:
    return [evaluate(probes, t, **kw)[0] for t in thresholds]


def grid(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 4) for i in range(n + 1)]


def choose_threshold(rows: list[SweepRow], max_false_hits: int = 0) -> SweepRow:
    """Most hits while false hits <= max_false_hits; ties go to the higher (safer) threshold."""
    ok = [r for r in rows if r.false_hits <= max_false_hits]
    if not ok:  # nothing is safe enough: take the fewest false hits
        return min(rows, key=lambda r: (r.false_hits, -r.hits))
    return max(ok, key=lambda r: (r.hits, r.threshold))


def exact_match_baseline(probes: list[Probe], cached_questions: list[str]) -> dict:
    """What a plain key-value cache (same text after normalising) would catch."""
    from faqcache.cache import normalize

    index = {normalize(q) for q in cached_questions}
    hits = [p for p in probes if normalize(p.item.question) in index]
    return {"hits": len(hits), "hit_rate": round(len(hits) / len(probes), 4)}


def score_bands(probes: list[Probe], width: float = 0.05) -> list[dict]:
    """Histogram of the top-1 similarity: 'should hit' vs 'should miss' questions."""
    pos = [p.candidates[0].score for p in probes if p.candidates and p.item.label != NONE_LABEL]
    neg = [p.candidates[0].score for p in probes if p.candidates and p.item.label == NONE_LABEL]
    edges = np.arange(0.0, 1.0001 + width, width)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        a = sum(lo <= s < hi for s in pos)
        b = sum(lo <= s < hi for s in neg)
        if a or b:
            out.append({"band": f"{lo:.2f}-{hi:.2f}", "should_hit": a, "should_miss": b})
    return out


def guard_ablation(probes, threshold, llm_latency_ms) -> dict:
    variants = {
        "no_guard": dict(negation_guard=False, fallthrough=False),
        "guard_skip": dict(negation_guard=True, fallthrough=False),
        "guard_fallthrough": dict(negation_guard=True, fallthrough=True),
    }
    results, details = {}, {}
    for name, kw in variants.items():
        row, rows = evaluate(probes, threshold, llm_latency_ms=llm_latency_ms, **kw)
        results[name] = asdict(row)
        details[name] = {r["id"]: r for r in rows}
    caught, cost, rescued = [], [], []
    for item_id, base in details["no_guard"].items():
        with_guard = details["guard_fallthrough"][item_id]
        if base["outcome"] == "FALSE_HIT" and with_guard["outcome"] != "FALSE_HIT":
            caught.append({**with_guard, "would_have_matched": base["matched"]})
        if base["outcome"] == "correct_hit" and with_guard["outcome"] != "correct_hit":
            cost.append(with_guard)
        if base["outcome"] != "correct_hit" and with_guard["outcome"] == "correct_hit":
            rescued.append(with_guard)
    return {"variants": results, "caught": caught, "lost": cost, "rescued": rescued}


# --------------------------------------------------------------------- end-to-end replay
def replay(bot_factory, items: list[TestItem]) -> dict:
    """Run every question through the real request path twice: without and with the cache."""

    def run(use_cache: bool) -> dict:
        bot = bot_factory()
        bot.cache_for()  # seed before timing
        lat, hits, tokens = [], 0, 0
        hit_ms, miss_lookup_ms, llm_ms = [], [], []
        start = time.perf_counter()
        for item in items:
            ans = bot.ask(item.question, use_cache=use_cache)
            lat.append(ans.latency_ms)
            hits += ans.source == "cache"
            tokens += ans.input_tokens + ans.output_tokens
            if ans.source == "cache":
                hit_ms.append(ans.latency_ms)
            else:
                llm_ms.append(ans.llm_ms)
                if use_cache:
                    miss_lookup_ms.append(ans.lookup_ms)

        def stats(xs):
            if not xs:
                return None
            return {
                "mean": round(statistics.fmean(xs), 2),
                "p50": round(statistics.median(xs), 2),
                "p95": round(float(np.percentile(xs, 95)), 2),
                "max": round(max(xs), 2),
            }

        return {
            "requests": len(items),
            "llm_calls": len(items) - hits,
            "cache_hits": hits,
            "tokens": tokens,
            "avg_latency_ms": round(statistics.fmean(lat), 2),
            "p50_latency_ms": round(statistics.median(lat), 2),
            "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
            "wall_s": round(time.perf_counter() - start, 2),
            "breakdown_ms": {
                "cache_hit_total": stats(hit_ms),
                "cache_lookup_on_miss": stats(miss_lookup_ms),
                "llm_call": stats(llm_ms),
            },
            "provider": bot.llm.name,
        }

    before, after = run(False), run(True)
    saved = 100 * (before["llm_calls"] - after["llm_calls"]) / before["llm_calls"]
    speed = 100 * (before["avg_latency_ms"] - after["avg_latency_ms"]) / before["avg_latency_ms"]
    return {
        "without_cache": before,
        "with_cache": after,
        "llm_calls_saved_pct": round(saved, 2),
        "avg_latency_reduction_pct": round(speed, 2),
    }

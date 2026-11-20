from collections import Counter

from faqcache import evaluation as ev
from faqcache.data import NONE_LABEL, TestItem, load_faqs, load_test_set
from faqcache.stores.base import CacheEntry, Candidate

from .conftest import ROOT


def test_dataset_shape():
    faqs = load_faqs(ROOT / "data" / "faqs.json")
    items = load_test_set(ROOT / "data" / "test_questions.jsonl", {f.id for f in faqs})
    assert len(faqs) == 30 and len(items) == 100
    assert sum(i.label != NONE_LABEL for i in items) == 60
    assert len({i.id for i in items}) == 100
    split = Counter((i.split, i.label != NONE_LABEL) for i in items)
    assert split[("dev", True)] == split[("test", True)] == 30
    assert split[("dev", False)] == split[("test", False)] == 20
    assert {i.lang for i in items} == {"en", "hinglish"}


def _probe(q, label, cands):
    item = TestItem(id=q, question=q, label=label, lang="en", kind="paraphrase")
    cs = [Candidate(CacheEntry(f, f, "a", 0, None, "faq", f), s) for f, s in cands]
    return ev.Probe(item, cs, lookup_ms=10.0)


def test_metrics_count_false_hits():
    probes = [
        _probe("a", "F01", [("F01", 0.9)]),  # correct hit
        _probe("b", "F02", [("F05", 0.9)]),  # wrong FAQ -> false hit
        _probe("c", NONE_LABEL, [("F03", 0.85)]),  # should miss -> false hit
        _probe("d", "F04", [("F04", 0.6)]),  # missed at 0.8
    ]
    row, _ = ev.evaluate(probes, 0.8, llm_latency_ms=100)
    assert (row.hits, row.correct_hits, row.false_hits) == (3, 1, 2)
    assert row.recall == round(1 / 3, 4) and row.llm_calls == 1
    assert row.est_avg_latency_ms == 10 + 100 / 4
    assert ev.evaluate(probes, 0.95)[0].hits == 0


def test_choose_threshold_prefers_zero_false_hits():
    rows = ev.sweep(
        [_probe("a", "F01", [("F01", 0.9)]), _probe("c", NONE_LABEL, [("F03", 0.85)])],
        [0.8, 0.86, 0.95],
    )
    assert ev.choose_threshold(rows).threshold == 0.86


def test_grid():
    assert ev.grid(0.75, 0.95, 0.05) == [0.75, 0.8, 0.85, 0.9, 0.95]

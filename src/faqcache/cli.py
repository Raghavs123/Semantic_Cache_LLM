"""Command line:  faqcache <command>   (or  python -m faqcache <command>)

ask "question"      one question through the full bot
chat                interactive chat in the terminal
demo                same question asked 3 ways: 1st is a miss (LLM), next ones hit the cache
eval                threshold sweep + dev/test hold-out + guard ablation -> reports/
replay              every test question end-to-end, without and with the cache -> reports/
serve               start the HTTP API (FastAPI + Uvicorn)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from faqcache.config import Settings
from faqcache.data import NONE_LABEL, load_faqs, load_test_set


def _settings(args) -> Settings:
    overrides = {}
    for key in ("threshold", "llm_provider", "store", "embedder", "embedding_model"):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    if getattr(args, "no_variants", False):
        overrides["seed_variants"] = False
    if getattr(args, "fake_latency_ms", None) is not None:
        overrides["fake_llm_latency_ms"] = args.fake_latency_ms
    return Settings(**overrides)


def _print_answer(ans) -> None:
    tag = "CACHE HIT " if ans.source == "cache" else "LLM (miss)"
    sim = "-" if ans.similarity is None else f"{ans.similarity:.3f}"
    print(f"[{tag}] {ans.latency_ms:8.1f} ms  sim={sim}  reason={ans.reason}")
    if ans.matched_question:
        print(f"           matched: {ans.matched_question}")
    print(f"           answer : {ans.answer}\n")


# ------------------------------------------------------------------------------ commands
def cmd_ask(args) -> int:
    from faqcache.factory import build_bot

    bot = build_bot(_settings(args))
    _print_answer(bot.ask(args.question))
    return 0


def cmd_chat(args) -> int:
    from faqcache.factory import build_bot

    bot = build_bot(_settings(args))
    print("DemoMart FAQ bot. Type a question (empty line to quit).\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        _print_answer(bot.ask(q))
    print(json.dumps(bot.logger.stats(), indent=2))
    return 0


# Similarity to the first question with the default model: 0.891 and 0.861 (measured).
# "Do you deliver on Sundays?" -> "Kya Sunday ko delivery hoti hai?" scored only 0.665,
# below T: a reminder that Hinglish recall is the weak spot.
DEMO_QUESTIONS = (
    "Can I pay with a credit card?",
    "Is credit card payment accepted?",
    "Credit card se payment ho sakta hai?",
)


def cmd_demo(args) -> int:
    from faqcache.factory import build_bot

    settings = _settings(args)
    bot = build_bot(settings, seed=False)  # start with an EMPTY cache
    print(f"provider={bot.llm.name}  threshold={settings.threshold}  (cache starts empty)\n")
    for q in args.questions or DEMO_QUESTIONS:
        print(f"Q: {q}")
        _print_answer(bot.ask(q))
    print(json.dumps(bot.logger.stats(), indent=2))
    return 0


def cmd_eval(args) -> int:
    from faqcache import evaluation as ev
    from faqcache import reports
    from faqcache.embeddings import build_embedder
    from faqcache.factory import build_cache

    s = _settings(args)
    faqs = load_faqs(s.faqs_path)
    items = load_test_set(s.test_set_path, {f.id for f in faqs})
    embedder = build_embedder(s.embedder, s.embedding_model)
    cache = build_cache(s.model_copy(update={"top_k": max(s.top_k, 5)}), embedder)
    n_entries = cache.seed(faqs, include_variants=s.seed_variants)
    llm_ms = args.llm_latency_ms

    probes = ev.probe_all(cache, items)
    lookup = sorted(p.lookup_ms for p in probes)
    spec_grid = ev.grid(0.75, 0.95, 0.05)
    fine_grid = ev.grid(0.50, 0.99, 0.01)
    kw = {"llm_latency_ms": llm_ms}

    spec_rows = ev.sweep(probes, spec_grid, **kw)
    spec_choice = ev.choose_threshold(spec_rows, args.max_false_hits)
    fine_rows = ev.sweep(probes, fine_grid, **kw)

    dev = [p for p in probes if p.item.split == "dev"]
    test = [p for p in probes if p.item.split == "test"]
    dev_rows = ev.sweep(dev, fine_grid, **kw)
    t_star = ev.choose_threshold(dev_rows, args.max_false_hits).threshold
    test_row, _ = ev.evaluate(test, t_star, **kw)
    full_row, per_question = ev.evaluate(probes, t_star, **kw)

    ablation = ev.guard_ablation(probes, t_star, llm_ms)
    bands = ev.score_bands(probes)
    cached_qs = [q for f in faqs for q in f.phrasings(s.seed_variants)]
    exact = ev.exact_match_baseline(probes, cached_qs)

    out = s.reports_dir
    out.mkdir(parents=True, exist_ok=True)
    reports.write_csv(out / "sweep_spec_grid.csv", spec_rows)
    reports.write_csv(out / "sweep_fine_grid.csv", fine_rows)
    summary = {
        "date": date.today().isoformat(),
        "embedding_model": embedder.name,
        "store": s.store,
        "seed_variants": s.seed_variants,
        "n_cache_entries": n_entries,
        "n_faqs": len(faqs),
        "n_questions": len(items),
        "n_should_hit": sum(i.label != NONE_LABEL for i in items),
        "chosen_threshold_dev": t_star,
        "chosen_threshold_spec_grid": spec_choice.threshold,
        "full_set_at_chosen": full_row.__dict__,
        "test_half_at_chosen": test_row.__dict__,
        "exact_match_baseline": exact,
        "lookup_latency_ms": {
            "median": round(lookup[len(lookup) // 2], 2),
            "p95": round(lookup[int(0.95 * (len(lookup) - 1))], 2),
            "max": round(lookup[-1], 2),
        },
        "llm_latency_ms_assumed_for_estimates": llm_ms,
        "guard_ablation": ablation["variants"],
    }
    reports.write_json(out / "summary.json", summary)
    reports.write_json(out / "per_question.json", per_question)

    fh = full_row
    md = [
        "# Evaluation report",
        "",
        f"Generated by `faqcache eval` on {summary['date']}. Embedding model: "
        f"`{embedder.name}`; store: `{s.store}`; {len(faqs)} FAQs cached as {n_entries} entries "
        f"(extra phrasings: {'on' if s.seed_variants else 'off'}); {len(items)} labelled "
        f"questions ({summary['n_should_hit']} should hit, "
        f"{len(items) - summary['n_should_hit']} should miss).",
        "",
        "Latency columns are *estimates*: measured cache-lookup time for every question, plus an "
        f"**assumed {llm_ms:.0f} ms** for each LLM call on a miss. Run `faqcache replay` with a "
        "real provider for measured end-to-end numbers.",
        "",
        "## 1. Threshold sweep, 0.75-0.95 step 0.05 (all 100 questions)",
        "",
        reports.sweep_table(spec_rows, spec_choice.threshold),
        f"Rule: most hits with false hits <= {args.max_false_hits}; ties -> higher threshold.",
        "",
        "## 2. Choosing the threshold without cheating (dev/test hold-out)",
        "",
        "The 100 questions are split 50/50 (stratified, seed 42). The threshold is picked on "
        "**dev** using a 0.01 grid, then reported on **test**, which it never saw.",
        "",
        f"* chosen on dev: **T = {t_star:.2f}**",
        f"* test half at T: hits {test_row.hits}/{test_row.n}, correct {test_row.correct_hits}, "
        f"false hits **{test_row.false_hits}**, recall {test_row.recall:.0%}",
        f"* all 100 at T: hit rate **{fh.hit_rate:.0%}** ({fh.hits}/100), correct "
        f"{fh.correct_hits}, false hits **{fh.false_hits}**, recall {fh.recall:.0%}, "
        f"LLM calls {fh.llm_calls} instead of 100",
        "",
        "## 3. Exact-match cache vs semantic cache",
        "",
        f"A plain exact-text cache would hit **{exact['hits']}** of 100 questions; the semantic "
        f"cache hits **{fh.correct_hits}** correctly at T = {t_star:.2f}.",
        "",
        "## 4. Similarity of the nearest FAQ (top-1 score)",
        "",
        reports.md_table(bands),
        "Where the two columns overlap, no single threshold can separate them perfectly.",
        "",
        f"## 5. Negation guard (at T = {t_star:.2f})",
        "",
        reports.md_table(
            [{"variant": k, **v} for k, v in ablation["variants"].items()],
            ["variant", "hits", "correct_hits", "false_hits", "recall", "precision"],
        ),
        "* `no_guard`: plain nearest neighbour. `guard_skip`: miss if the best match is blocked. "
        "`guard_fallthrough`: try the next candidate (top-k) if the best one is blocked.",
        "",
        "**False hits the guard caught:**",
        "",
        reports.md_table(ablation["caught"], ["id", "question", "would_have_matched", "outcome"]),
        "**Correct hits the guard cost us (honest downside):**",
        "",
        reports.md_table(ablation["lost"], ["id", "question", "label", "reason"]),
        "**Questions the fall-through rescued (blocked top-1, correct 2nd candidate):**",
        "",
        reports.md_table(ablation["rescued"], ["id", "question", "matched", "score"]),
        "## 6. Every remaining false hit and missed paraphrase at T",
        "",
        reports.md_table(
            [r for r in per_question if r["outcome"] in ("FALSE_HIT", "missed")],
            ["id", "question", "label", "matched", "score", "reason", "outcome"],
        ),
        "## 7. Lookup latency (measured, this machine)",
        "",
        f"Median {summary['lookup_latency_ms']['median']} ms, p95 "
        f"{summary['lookup_latency_ms']['p95']} ms, max {summary['lookup_latency_ms']['max']} ms "
        "per lookup (normalise + embed on CPU + search + guards), single process.",
        "",
    ]
    (out / "EVAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")

    print(reports.sweep_table(spec_rows, spec_choice.threshold))
    print(
        f"dev-chosen T={t_star:.2f} | all 100: hit rate {fh.hit_rate:.0%}, false hits "
        f"{fh.false_hits}, recall {fh.recall:.0%} | test half: false hits {test_row.false_hits}"
    )
    print(f"exact-match baseline hits: {exact['hits']}   reports written to {out}/")
    return 0


DEFAULT_COMPARE_MODELS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "intfloat/multilingual-e5-small",
)


def cmd_compare(args) -> int:
    """Same data, same rules, different embedding models / seeding: one row each."""
    from faqcache import evaluation as ev
    from faqcache import reports
    from faqcache.embeddings import build_embedder
    from faqcache.factory import build_cache

    s = _settings(args)
    faqs = load_faqs(s.faqs_path)
    items = load_test_set(s.test_set_path, {f.id for f in faqs})
    fine_grid = ev.grid(0.50, 0.99, 0.01)
    rows = []
    for model in args.models or DEFAULT_COMPARE_MODELS:
        embedder = build_embedder("sentence-transformers", model)
        for variants in (False, True):
            cache = build_cache(s.model_copy(update={"top_k": 5}), embedder)
            cache.seed(faqs, include_variants=variants)
            probes = ev.probe_all(cache, items)
            dev = [p for p in probes if p.item.split == "dev"]
            test = [p for p in probes if p.item.split == "test"]
            t = ev.choose_threshold(ev.sweep(dev, fine_grid), args.max_false_hits).threshold
            full, _ = ev.evaluate(probes, t)
            held, _ = ev.evaluate(test, t)
            at80, _ = ev.evaluate(probes, 0.80)
            lat = sorted(p.lookup_ms for p in probes)
            rows.append(
                {
                    "model": model.split("/")[-1],
                    "extra_phrasings": "on" if variants else "off",
                    "T_dev": t,
                    "hits@T": full.hits,
                    "correct@T": full.correct_hits,
                    "false@T": full.false_hits,
                    "test_half_correct": f"{held.correct_hits}/30",
                    "test_half_false": held.false_hits,
                    "correct@0.80": at80.correct_hits,
                    "false@0.80": at80.false_hits,
                    "median_lookup_ms": round(lat[len(lat) // 2], 1),
                }
            )
            print(rows[-1], flush=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    reports.write_json(s.reports_dir / "model_comparison.json", rows)
    md = [
        "# Embedding model comparison",
        "",
        f"Generated by `faqcache compare` on {date.today().isoformat()}. Same 30 FAQs, same 100 "
        "questions, same guards. T_dev = threshold picked on the dev half (most hits with "
        f"false hits <= {args.max_false_hits}); `test_half_*` = the 50 questions it never saw "
        "(30 should hit). `@0.80` columns use one fixed threshold for every model, to show that "
        "a threshold is NOT portable between models. Lookup time = CPU, this machine.",
        "",
        reports.md_table(rows),
    ]
    (s.reports_dir / "MODEL_COMPARISON.md").write_text("\n".join(md), encoding="utf-8")
    print(reports.md_table(rows))
    return 0


def cmd_replay(args) -> int:
    from faqcache import evaluation as ev
    from faqcache import reports
    from faqcache.embeddings import build_embedder
    from faqcache.factory import build_bot

    s = _settings(args)
    if args.threshold is None:
        summary = s.reports_dir / "summary.json"
        if summary.exists():
            t = json.loads(summary.read_text(encoding="utf-8"))["chosen_threshold_dev"]
            s = s.model_copy(update={"threshold": t})
    s = s.model_copy(update={"log_path": None})
    faqs = load_faqs(s.faqs_path)
    items = load_test_set(s.test_set_path, {f.id for f in faqs})[: args.limit]
    embedder = build_embedder(s.embedder, s.embedding_model)
    result = ev.replay(lambda: build_bot(s, embedder=embedder, faqs=faqs), items)
    result["threshold"] = s.threshold
    result["simulated_llm"] = s.llm_provider == "fake"
    if result["simulated_llm"]:
        result["note"] = (
            f"LLM was the offline stand-in sleeping {s.fake_llm_latency_ms:.0f} ms per call: "
            "LLM-side latency is SIMULATED; cache-side latency is measured."
        )
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    reports.write_json(s.reports_dir / "replay.json", result)
    print(json.dumps(result, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("faqcache.api:app", host=args.host, port=args.port)
    return 0


# ------------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="faqcache", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--threshold", type=float)
        sp.add_argument("--llm-provider", dest="llm_provider")
        sp.add_argument("--store", choices=["memory", "faiss"])
        sp.add_argument("--embedder", choices=["sentence-transformers", "hash"])
        sp.add_argument("--fake-latency-ms", dest="fake_latency_ms", type=float)
        sp.add_argument("--model", dest="embedding_model", help="embedding model name")
        sp.add_argument(
            "--no-variants", action="store_true", help="cache only the main FAQ question"
        )
        return sp

    sp = common(sub.add_parser("ask", help="ask one question"))
    sp.add_argument("question")
    sp.set_defaults(fn=cmd_ask)

    common(sub.add_parser("chat", help="interactive chat")).set_defaults(fn=cmd_chat)

    sp = common(sub.add_parser("demo", help="3 phrasings, empty cache"))
    sp.add_argument("questions", nargs="*")
    sp.set_defaults(fn=cmd_demo)

    sp = common(sub.add_parser("eval", help="sweep + hold-out + guard ablation"))
    sp.add_argument("--max-false-hits", type=int, default=0)
    sp.add_argument("--llm-latency-ms", type=float, default=800.0)
    sp.set_defaults(fn=cmd_eval)

    sp = common(sub.add_parser("compare", help="compare embedding models"))
    sp.add_argument("models", nargs="*")
    sp.add_argument("--max-false-hits", type=int, default=0)
    sp.set_defaults(fn=cmd_compare)

    sp = common(sub.add_parser("replay", help="end-to-end, without vs with cache"))
    sp.add_argument("--limit", type=int, default=None)
    sp.set_defaults(fn=cmd_replay)

    sp = sub.add_parser("serve", help="run the HTTP API")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.fn(args)

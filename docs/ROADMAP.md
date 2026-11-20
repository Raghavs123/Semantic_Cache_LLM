# 3-day build roadmap: Semantic cache for an FAQ bot

> How to read this: each day has numbered steps. Each step says **what** to build, **why**,
> **which file**, the **command** to check it, and a **done when** line. The finished code is
> in this repo, so you can read along. You learn far more if you type each file yourself
> first and only look here when you are stuck.

---

## Day 0 (30 minutes): the idea in plain words

**Life example.** A shop helper gets asked "where is my order?" 200 times a day. After the
first time, she writes the answer on a sticky note. When someone asks "mera order kahan hai?"
she sees it *means the same thing* and reads the note. That's much faster than phoning the
head office (the LLM) every time.

* **Exact cache** = the note only works when the words match exactly. "Where is my order?"
  and "where's my order" count as different questions. Almost useless for real chat.
* **Semantic cache** = turn every question into a list of numbers (an *embedding*) that
  captures its meaning. Questions with similar meanings get similar numbers. If the new
  question is close enough to a saved one, reuse that saved answer.
* **Threshold** = how close counts as "the same question". Too low gives wrong answers
  (*false hits*); too high means the cache almost never answers.
* **The proof** = a labelled test set, so you can *measure* the hit rate and false hits
  instead of guessing.

```
question --> [L1 exact match?] --yes--> saved answer (about 0 ms)
                 | no
                 v
            embed question (about 10-15 ms on a CPU)
                 |
            nearest saved questions (top-k) --> similarity >= T ?  --> guards OK? --> HIT
                 |                                                        |
                 +------------------------- MISS <------------------------+
                                             |
                                 ask the LLM (hundreds of ms) --> save Q+A (if safe) --> answer
```

---

## Folder structure (industry-style "src layout")

```
semantic-faq-cache/
├── pyproject.toml            # one file for dependencies, tools, entry points (PEP 621)
├── README.md                 # the résumé page: diagram, numbers, how to run
├── .env.example              # every setting, with safe defaults (copy to .env)
├── Dockerfile / docker-compose.yml
├── .github/workflows/ci.yml  # lint + tests on every push
├── data/
│   ├── faqs.json             # 30 FAQs (+2 extra phrasings each)
│   └── test_questions.jsonl  # 100 labelled questions (60 should hit, 40 should miss)
├── src/faqcache/
│   ├── config.py             # Settings (pydantic-settings, FAQ_* env vars)
│   ├── data.py               # load + validate data, dev/test split
│   ├── embeddings.py         # Embedder interface, real model, offline hash model
│   ├── guards.py             # negation guard, personal-data guard, TTL policy
│   ├── stores/               # vector stores: memory (NumPy), FAISS, Redis template
│   ├── cache.py              # SemanticCache + the pure decide() function
│   ├── llm/                  # LLM interface + fake / Anthropic / OpenAI / Gemini / Ollama
│   ├── service.py            # FAQBot: cache -> LLM -> save -> log; one cache per tenant
│   ├── factory.py            # builds everything from Settings (composition root)
│   ├── api.py                # FastAPI: POST /v1/ask, GET /v1/stats, GET /healthz
│   ├── evaluation.py         # sweep, hold-out, guard ablation, replay
│   ├── reports.py            # markdown / CSV / JSON writers
│   └── cli.py                # `faqcache ask|chat|demo|eval|compare|replay|serve`
├── tests/                    # pytest, offline (no model download, no API key)
├── reports/                  # generated numbers (committed, so the README can cite them)
└── docs/ROADMAP.md           # this file
```

**Why this shape?** Each file has one job, and the parts talk through small interfaces
(`Embedder`, `VectorStore`, `LLMClient`). You can swap NumPy for FAISS, or Claude for Ollama,
without touching the cache logic. Interviewers call this *separation of concerns* and
*dependency injection*. `src/` layout stops tests from accidentally importing local files
instead of the installed package.

---

## Day 1: build it (about 6-7 hours)

### Step 1.1: set up the project (30 min)
* Create the folders above and `pyproject.toml` (copy ours and read every line).
* Create a virtual environment and install in *editable* mode:
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
  pip install -e ".[dev,faiss]"
  ```
* `git init`, add `.gitignore`, commit "skeleton".
* **Done when:** `python -c "import faqcache; print(faqcache.__version__)"` prints `1.0.0`.

### Step 1.2: write the data (60-90 min). This is the most important hour.
* `data/faqs.json`: pick one domain (we use a made-up shop, *DemoMart*). Write 30 FAQs with
  `id`, `question`, `answer`, `category`. Keep answers short and **generic**: no names and no
  order numbers.
* `data/test_questions.jsonl`: 100 lines. 60 **paraphrases** (2 per FAQ: one English, one
  Hinglish) labelled with the FAQ id. 40 that must **not** hit, labelled `"none"`:
  * 16 *out of scope* but shop-like ("Do you sell medicines?")
  * 8 *off topic* ("Who won the cricket match?")
  * 8 **hard negatives**: same words, different question ("How do I reset my **UPI PIN**?"
    vs the FAQ "How do I reset my **password**?")
  * 8 **negation flips** ("Why **can't** I cancel my order?" vs "How do I cancel my order?")
* **Rule:** write the test questions *before* you look at any similarity scores, or you will
  bend the test toward your system without noticing.
* **Done when:** `pytest tests/test_evaluation_and_data.py` passes (it checks 30/100/60/40).

### Step 1.3: settings (20 min). File `config.py`
* One `Settings` class (pydantic-settings) reads `FAQ_THRESHOLD`, `FAQ_LLM_PROVIDER`, …
  from the environment or `.env`. No magic numbers scattered through the code.
* API keys are *not* settings: each SDK reads its own variable (`ANTHROPIC_API_KEY`…).

### Step 1.4: embeddings (40 min). File `embeddings.py`
* An `Embedder` *Protocol* with `embed(texts) -> (n, dim) float32` array.
* `SentenceTransformerEmbedder` uses `normalize_embeddings=True`. Every vector then has
  length 1, so **dot product = cosine similarity**. Search becomes one matrix multiply.
* Model: start with `paraphrase-multilingual-MiniLM-L12-v2` (English + Hindi, small), or
  `all-MiniLM-L6-v2` if you only have English. On Day 2, `faqcache compare` showed that
  `paraphrase-multilingual-mpnet-base-v2` gave the best *safe* result on this data, so it is
  the repo's default now. Let measurements choose the model, not habit.
* `HashEmbedder`: a tiny offline fake for unit tests (spelling only, no meaning).
* **Check:**
  ```python
  from faqcache.embeddings import SentenceTransformerEmbedder
  e = SentenceTransformerEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  v = e.embed(["How long does a refund take?", "Refund kab milega?", "Capital of Australia?"])
  print(v @ v.T)   # row 0 vs 1 should be higher than row 0 vs 2
  ```

### Step 1.5: vector store (40 min). Folder `stores/`
* `CacheEntry` (id, question, answer, created_at, expires_at, source, faq_id) and
  `Candidate` (entry + score).
* `InMemoryVectorStore`: keep vectors in a NumPy matrix; `scores = M @ q`; take the top-k.
  Exact search, fine up to tens of thousands of entries.
* `FaissVectorStore`: the same thing with `faiss.IndexFlatIP` (inner product).
  `test_faiss_store_matches_memory_store` proves both return the same answers.

### Step 1.6: the cache (90 min). File `cache.py`
* `decide(question, candidates, threshold)` is a **pure function** (no I/O). It walks the
  candidates best-first: below the threshold → miss; blocked by a guard → skip (or try the
  next one); otherwise → hit. Keep it pure so the evaluation reuses *exactly* this logic.
* `SemanticCache.lookup()`: L1 exact dictionary (normalised text) → embed → `store.search`
  → drop expired entries → `decide()`. Returns hit/miss, reason, score and latency.
* `put()`: refuse personal data, compute TTL, store vector + entry.
* `seed(faqs)`: embed all 30 FAQ questions in **one batch** (much faster than one by one).

### Step 1.7: LLM providers (40 min). Folder `llm/`
* Interface `generate(question, system) -> LLMResponse(text, input_tokens, output_tokens,
  model, cacheable)`.
* `FakeLLM` first (no key, sleeps N ms), then real ones: `anthropic_client.py`,
  `openai_client.py`, `gemini_client.py`, `ollama_client.py`.
* `cacheable=False` for refusals or cut-off answers. **Never cache a bad answer**: the cache
  would repeat it for everyone.

### Step 1.8: the bot + logging (40 min). File `service.py`
* `FAQBot.ask()`: personal data? → skip the cache. Otherwise `lookup` → hit: return the saved
  answer. Miss: call the LLM → `put` (if cacheable) → return.
* `RequestLogger` writes one JSON line per request: question (**personal data masked**),
  hit/miss, reason, similarity, latency ms, tokens.
* **Done when:**
  ```bash
  faqcache ask "refund kab milega?"
  faqcache demo
  ```
  Expected: first question → `LLM (miss)`, the repeats → `CACHE HIT` in a few ms.

### Step 1.9: tests for Day 1 (30 min)
* `pytest` should pass offline: exact hit, semantic hit, miss, threshold boundary, user-data
  refusal, duplicate, unit-length vectors, FAISS parity.
* Commit: "day 1: working semantic cache".

---

## Day 2: tune and measure (about 6-7 hours)

### Step 2.1: probe once, sweep many times (60 min). File `evaluation.py`
* Seed the cache with the FAQs. Look up each of the 100 questions **once** and keep its
  top-k candidates plus the measured lookup time (`probe_all`).
* For each threshold, re-run `decide()` on the stored candidates. No re-embedding needed,
  so a 50-point sweep takes milliseconds.
* Count, per threshold: hits, **correct hits**, **false hits** (a "none" question got an
  answer, *or* the wrong FAQ was returned), recall (correct / 60), LLM calls, estimated
  average latency (lookup + assumed LLM time on a miss).

### Step 2.2: run the sweep (30 min)
```bash
faqcache eval          # writes reports/EVAL_REPORT.md, sweep_*.csv, summary.json
```
* Section 1 of the report is the 0.75-0.95 table from the project brief.
* **Read section 4 (score bands)**. If the "should hit" and "should miss" columns overlap,
  no threshold can be perfect, and you can explain that in an interview.

### Step 2.3: choose the threshold honestly (30 min)
* If you pick T on the same 100 questions you report on, you are grading your own exam.
* `data.assign_splits` makes a stratified 50/50 **dev/test** split. Pick T on dev (rule:
  most hits with zero false hits; ties → the higher, safer T), then report it on test.

### Step 2.4: negation guard (60 min). File `guards.py`
* `has_negation()` looks for not / no / never / n't / nahi / nahin / nhi / mat / without /
  bina, but ignores phrases like "no-cost EMI". `negation_mismatch(a, b)` is true when
  exactly one side is negated.
* In `decide()`: if the best match is blocked, either **skip** (miss) or **fall through**
  to the next candidate above T (top-k).
* The report's section 5 compares no guard / skip / fall-through, and lists which false hits
  the guard **caught** and which correct hits it **cost**. Keep 2-3 caught examples for the
  README.

### Step 2.5: cache hygiene (45 min)
* **TTL:** answers that mention price / fee / charge / offer / policy / EMI expire after
  `FAQ_VOLATILE_TTL_DAYS` (default 7); others never expire. Expired entries are evicted
  lazily on lookup (and by `evict_expired()`). Tested with a fake clock.
* **Personal data:** regexes for order ids, phone numbers, e-mails, "my name is…" and
  greetings like "Hi Rahul,". Such questions skip the cache completely, and such answers are
  never stored. Logs are masked with `redact()`.

### Step 2.6: improve, then re-measure (90 min)
Change **one** thing at a time and re-run `faqcache eval` after each:
1. **Extra phrasings**: each FAQ gets 2 more ways to ask (one Hinglish) in `faqs.json`
   (`variants`). Each one becomes its own cache entry with the same answer. Never copy a
   test question into the variants: that is leakage.
2. **Embedding model**: `faqcache compare` runs the same evaluation for several models,
   with phrasings on and off → `reports/MODEL_COMPARISON.md`. Note that the best threshold
   **differs per model**: a threshold is not portable.
* What we actually found (19 Nov 2026, see the README): the brief's 0.75-0.95 range was too
  high for some models and too low for others (e5-small: 57 false hits at 0.80). Romanised
  Hinglish scored low on every model. The negation guard turned 5 false hits into 0 at
  T = 0.83 without losing a correct hit.
* Commit: "day 2: measured + tuned".

---

## Day 3: ship it (about 5-6 hours)

### Step 3.1: HTTP API (45 min). File `api.py`
* FastAPI app factory `create_app(bot)`. The `lifespan` builds the bot once at start-up
  (loads the model, seeds the cache). `POST /v1/ask`, `GET /v1/stats`, `GET /healthz`.
  Input is validated by Pydantic (empty question → 422).
* `faqcache serve` → open http://127.0.0.1:8000/docs and try it.

### Step 3.2: end-to-end replay (30 min)
```bash
faqcache replay                                   # offline stand-in LLM (latency simulated)
FAQ_LLM_PROVIDER=anthropic faqcache replay        # real LLM: real latency and tokens
```
It runs every question through the real request path twice (cache off, then on) and
writes `reports/replay.json`: LLM calls, tokens, average / p50 / p95 latency, plus a breakdown
(hit time, cache time on a miss, LLM time). **Lesson we learned the hard way:** the first
version embedded every miss twice (lookup + put). With CPU jitter that made the cached run
*slower* on average. Always measure the miss path, not just the hit path.

### Step 3.3: packaging and CI (45 min)
* `Dockerfile` (CPU torch, model downloaded at build time, non-root user, healthcheck),
  `docker-compose.yml` (API + Redis Stack for the RedisVL option).
* `.github/workflows/ci.yml`: ruff lint + format check + pytest on Python 3.11-3.13.
* `.pre-commit-config.yaml` runs ruff before every commit.

### Step 3.4: README (60 min)
* Diagram, the threshold table, before/after numbers, how to run, design decisions,
  limitations. Fill the **one résumé line** with *your* measured numbers. Never round up.

### Step 3.5: demo video (30 min)
* Screen-record `faqcache demo` (or `/docs` in the browser). The first question is a miss
  that goes to the LLM; two other phrasings come back from the cache in milliseconds.
  Keep it under 60 seconds.

### Step 3.6: interview prep (30 min)
See "Talking points" in the README. Practise saying each one out loud in 30 seconds.

---

## Stretch goals (after the 3 days)
* Redis backend from `stores/redis_template.py` (shared cache across many API workers).
* A small cross-encoder **re-ranker** on the top-3 candidates before accepting a hit.
* An **entity guard**: numbers, product names or dates must match ("7 days" ≠ "30 days").
* Per-category thresholds (payments stricter than delivery).
* A feedback button: a 👎 on a cached answer evicts it.
* Metrics with Prometheus (`hit_rate`, `false_hit_reports`, p95 latency) plus a Grafana panel.

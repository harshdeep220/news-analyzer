# Real-Time News RAG — Multi-Agent Research & Credibility System
## Architecture & Build Roadmap — Version 2.0
### Stack: Google AI Studio · Flask · ChromaDB · LangGraph · Tavily

> **This document is the single source of truth. Read it completely before writing any code. Reference it daily. Never start Phase N+1 without passing the smoke test at the end of Phase N. There is no restart — every decision here is load-bearing.**

---

## Label Legend

| Label | Meaning |
|-------|---------|
| ✅ COMPLETE | Phase or task done and smoke-tested |
| ▶ NEXT | Immediate next task |
| ⏳ PENDING | Not yet started |
| 🧪 SMOKE TEST | End-of-phase integration test — must pass before proceeding |
| ⚠ RISK | Known failure mode |
| ✔ FIX | Concrete solution to the risk above it |
| ❌ REMOVED | Was in v1.0 — explicitly cut in v2.0 and why |

---

## What Changed From v1.0 — And Why

This section exists so you never wonder "why isn't X here." Every removal was deliberate.

| v1.0 Component | v2.0 Replacement | Reason |
|---|---|---|
| Groq API (5 keys, GroqKeyPool, token-bucket rotation) | Google AI Studio (1 key, 1 client) | GroqKeyPool was the single biggest complexity source and the most common failure point. Multi-key rotation, per-key RPM buckets, and load balancing logic added ~2 days of infrastructure work that produced zero user-facing value. One key eliminates all of it. |
| `llama-3.1-8b-instant` | `gemini-2.0-flash` | Direct replacement for all fast tasks. Faster, higher quality, same API surface. |
| `llama-3.3-70b-versatile` | `gemini-2.0-flash` | Flash handles Source Comparator and Critic at this task complexity. No need for a separate mid-tier model. |
| `openai/gpt-oss-120b` (synthesis) | `gemini-2.5-pro` | Superior reasoning. Thinking tokens are separated by the API natively — no `clean_reasoning_output()` function needed. |
| `openai/gpt-oss-20b` (fast path) | `gemini-2.0-flash` | Consolidated. Flash is fast enough for all non-synthesis tasks. |
| Ollama + `embedding-gemma` (local GPU) | `text-embedding-004` via google-genai SDK | Eliminates local GPU dependency entirely. No `ollama list`, no health-check, no 4GB model pull, no startup verification. Same API call as the LLM. |
| Stitch (Google, beta design tool) + HTMX | Plain Flask templates + native `EventSource` JS | Stitch is a beta tool with unpredictable export quality. HTMX SSE requires a separate extension JS file. Plain `EventSource` is 30 lines of standard JavaScript, zero dependencies, and does the same thing. |
| `groq_client.py` (GroqKeyPool class) | `google_client.py` (thin wrapper) | One file, one class, two methods: `generate()` and `embed()`. |
| `gemma_embedder.py` | Folded into `google_client.py` | Embeddings and LLM calls are the same SDK. No separate file needed. |
| `deepseek-r1` reasoning token stripping | Nothing | Not needed. Gemini 2.5 Pro does not leak thinking tokens into responses. |

**Nothing else changed.** Every agent, every pipeline node, every RAG technique, every session management decision, every smoke test requirement — all preserved exactly.

---

## Confirmed Google AI Studio Models

| Model ID | Speed | Context | Role |
|----------|-------|---------|------|
| `gemini-2.0-flash` | Fast | 1M tokens | Intent, CRAG, Bias, Credibility, Formatter, Critic, Source Comparator, Hallucination Check |
| `gemini-2.5-pro` | Slower | 1M tokens | Synthesis + deep reasoning only |
| `text-embedding-004` | Fast | 2048 tokens input | All vector embeddings — 768-dim output |

> **Rate limits (free tier):** Flash = 15 RPM / 1M TPM. Pro = 2 RPM / 32K TPM. These limits make the QueryCache (Day 5) non-negotiable during development — repeated test queries will hit Pro limits almost immediately without it.

> **One config file rule:** `config.py` defines `FAST_MODEL = 'gemini-2.0-flash'`, `SYNTH_MODEL = 'gemini-2.5-pro'`, `EMBED_MODEL = 'text-embedding-004'`. No model string appears anywhere else in the codebase. Ever.

---

## System Architecture — Full Pipeline

```
USER MESSAGE (session_id attached)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                   SESSION MANAGER                   │
│  • SQLite lookup: session_id → safe_collection_name │
│  • Load or create ChromaDB collection               │
│  • Load message history from SQLite                 │
│  • Trigger history compression if msg_count > 10   │
│    (async daemon thread, threading.Lock on writes)  │
│  • Auto-label topic when word_count > 100           │
│  • QueryCache check: (session_id, query_hash) hit?  │
│    → return cached response immediately             │
└─────────────────────────────────────────────────────┘
         │ context bundle
         ▼
┌─────────────────────────────────────────────────────┐
│         ORCHESTRATOR [LangGraph StateGraph]         │
│         Global timeout: 60 seconds hard abort       │
│                                                     │
│  NODE 1: INTENT + QUERY PLANNER [Flash]             │
│  Single combined call → {intent, sub_queries,       │
│  search_needed, entity_heavy}                       │
│  History summary: max 500 words passed in           │
│  chitchat → respond directly, END                   │
│  follow-up → RAG-first (skip Tavily if cache hot)   │
│  new_research → full pipeline                       │
│                                                     │
│  NODE 2: PARALLEL DISPATCH                          │
│  ├── RAG Agent (BM25 + Semantic + RRF + Reranker)  │
│  └── Web Search Agent (Tavily + dedup validator)    │
│                                                     │
│  NODE 3: DATA VALIDATION LAYER                      │
│  • length > 200 chars, not paywalled, not duplicate │
│  • wire-story hash dedup (MD5, first 300 chars)     │
│  • future date cap → set to today, log warning      │
│  Validated chunks → embed → session ChromaDB        │
│  (ingested_at timestamp stored as metadata)         │
│                                                     │
│  NODE 4: CRAG GRADER [Flash]                        │
│  Grade each doc: relevant / partial / irrelevant    │
│  Entire batch in ONE API call                       │
│  All irrelevant → Tavily-only (skip RAG)            │
│  New session (< 5 docs) → Tavily-only automatically │
│                                                     │
│  NODE 5: CREDIBILITY PIPELINE [Flash]               │
│  Signal 1: source_tier (local JSON DB, 35%)         │
│  Signal 2: cross-corroboration score (30%)          │
│    Wire-republisher weight: 0.5x per duplicate      │
│  Signal 3: content quality LLM grade (20%)          │
│  Signal 4: freshness decay (15%)                    │
│  → credibility_score 0-100 per source               │
│  Cache by URL in SQLite (never re-score same URL)   │
│                                                     │
│  NODE 6: BIAS DETECTOR [Flash]                      │
│  Skip if article < 250 words → insufficient_content │
│  → {lean, confidence, bias_types, loaded_language,  │
│     missing_perspectives}                           │
│                                                     │
│  NODE 7: SOURCE COMPARATOR [Flash]                  │
│  Trigger: 2+ articles share ≥3 of top-5 entity      │
│  keywords AND published within 24h                  │
│  (cheap keyword check — no extra API call)          │
│  → {agreed_facts, disputed_facts, unique_claims,    │
│     framing_differences}                            │
│  First 800 words per article passed to comparator   │
│                                                     │
│  NODE 8: SYNTHESIS [gemini-2.5-pro]                 │
│  Iterative RAG: up to 2 passes if gaps detected     │
│  Gap must be > 5 words to justify a new pass        │
│  Stores synthesis_attempts[] for circuit breaker    │
│  Thinking tokens handled natively — no stripping    │
│                                                     │
│  NODE 9: HALLUCINATION CHECK [Flash]                │
│  Is synthesis grounded in retrieved docs?           │
│  Ungrounded claims → flag with ⚠, never hidden      │
│  grounding_score fed into credibility display       │
│                                                     │
│  NODE 10: CRITIC [Flash]                            │
│  Quality gate: completeness, accuracy, coverage     │
│  CIRCUIT BREAKER: max 2 retries                     │
│  Pass threshold: score > 0.4 regardless of flag     │
│  On exhaustion: return max(attempts, key=score)     │
│  NOT the last attempt                               │
│                                                     │
│  NODE 11: FORMATTER [Flash]                         │
│  Citations + hyperlinks + credibility badges        │
│  Bias indicators per source                         │
│  Explainable retrieval panel                        │
│  All HTML validated through html.parser             │
│  Plain-text fallback on parse error                 │
└─────────────────────────────────────────────────────┘
         │
         ▼
   Final Response (SSE stream)
         │
   QueryCache.store(session_id, query_hash, response)
         │
         ▼
   FLASK UI (Jinja2 templates + EventSource JS)
```

---

## RAG Techniques

| # | Technique | Where Used | Why |
|---|-----------|------------|-----|
| 1 | Fusion Retrieval (BM25 + Semantic + RRF) | NODE 2 — RAG Agent | News has named entities. BM25 catches exact names; semantic catches meaning. Both required. |
| 2 | Contextual Chunk Headers | Ingestion — every embed | Prepend `[Source | Date | Outlet]` before embedding. Source filtering at embedding level, not just metadata. |
| 3 | Cross-Encoder Reranking (ms-marco-MiniLM) | NODE 2 — post-retrieval | Top-10 → top-5. Runs on CPU locally. Default top_k=10, not 20 — saves 500ms+ with minimal quality loss. |
| 4 | Multi-faceted Filtering | NODE 2 — pre-retrieval | Filter by `published_date` recency + `source_tier` before vector search. |
| 5 | Corrective RAG (CRAG) | NODE 4 — grader | Grades docs relevant/partial/irrelevant. Routes to Tavily if all docs score low. Whole batch = one API call. |
| 6 | Adaptive Retrieval | NODE 1 — intent classifier | follow-up → RAG-only; new_research → full pipeline; cold session → Tavily-only. |
| 7 | Iterative Retrieval | NODE 8 — synthesis | Up to 2 retrieval passes. Gap must be > 5 words to justify a pass. |
| 8 | Reliable RAG (Hallucination Grading) | NODE 9 — grounding check | Is synthesis grounded in retrieved docs? Binary per claim. |
| 9 | Explainable Retrieval | NODE 11 — formatter | Show exactly which chunks each claim came from. |

**Explicitly excluded (reasons unchanged from v1.0):** HyDE, RAPTOR, GraphRAG, HyPE, Proposition Chunking, Semantic Chunking.

---

## Known Issues Registry — All Addressed

### Issue #1: Google AI Studio Rate Limits
**Fix location:** `google_client.py`, `infrastructure/query_cache.py`

**⚠ RISK:** Flash = 15 RPM free tier. Pro = 2 RPM. With 6–8 calls per new_research query during development, hitting Pro limits is nearly instant without caching.

**✔ FIX:** `QueryCache` class: key = `sha256(session_id + query)`, TTL = 60 seconds, in-memory dict. Check cache before entering the LangGraph graph — a cache hit never touches any model. Additionally: `tenacity` exponential backoff (min=2s, max=30s, max_attempts=4) on every API call. In tests, use the Tavily fixture and mock the LLM calls — never make live API calls in unit tests.

---

### Issue #2: ChromaDB Collection Naming
**Fix location:** `session/session_manager.py`

**⚠ RISK:** ChromaDB: 3–63 chars, alphanumeric+hyphens only, must start/end with alphanumeric, no consecutive hyphens. UUID4 strings fail these rules.

**✔ FIX:** `_safe_collection_name()`: strips non-alphanumeric, prefixes with `s`, clamps to 63 chars. Hash fallback if result < 3 chars. Store `session_id → collection_name` in SQLite and ALWAYS look up — never reconstruct. `validate_all_collections()` at startup logs orphaned sessions, never crashes.

---

### Issue #3: ChromaDB Session Collection Grows Unbounded
**Fix location:** `session/session_manager.py`

**⚠ RISK:** After ~300 chunks, retrieval quality degrades and query latency increases.

**✔ FIX:** `MAX_CHUNKS_PER_SESSION = 300`. Before each `add_to_collection()`, check `collection.count()`. If at limit, query oldest 50 chunks by **`ingested_at` metadata** (not `published_date` — future-date-capped articles would otherwise never be evicted) and delete before inserting new ones.

---

### Issue #4: Tavily Wire Story Duplicates
**Fix location:** `pipeline/data_validator.py`

**⚠ RISK:** AP/Reuters wire stories republished identically by 50+ outlets. All return from Tavily as separate results. Inflates corroboration scores artificially.

**✔ FIX:** `is_wire_duplicate()`: MD5 hash of first 300 chars (lowercased, stripped). `seen_hashes` set passed as parameter to `validate_batch(articles, seen_hashes=None)` — per-batch scope, not global. First occurrence passes; duplicates dropped.

---

### Issue #5: Tavily Future Date Metadata
**Fix location:** `agents/web_search_agent.py`

**⚠ RISK:** Tavily sometimes returns articles with future dates. These score 0 on freshness decay and corrupt temporal filtering.

**✔ FIX:** Cap `published_date` at today. If article date > today, set to today. Log as `metadata_warning` in Article dataclass. Use `ingested_at` (UTC time of embedding) for eviction ordering — never `published_date`.

---

### Issue #6: Bias Detector Hallucination on Short Articles
**Fix location:** `pipeline/bias_detector.py`

**⚠ RISK:** 100–150 word article stubs produce confident political lean classifications from nothing.

**✔ FIX:** `if len(content.split()) < 250: return BiasResult(lean='insufficient_content', confidence=0.0)`. Zero API calls. Grey badge in UI.

---

### Issue #7: Topic Label Triggers on Single-Word Replies
**Fix location:** `session/topic_labeler.py`

**⚠ RISK:** Labelling after message #2 captures `yes` or `tell me more`.

**✔ FIX:** Trigger only when total word count across session exceeds 100 words AND label is still `New Research`. Never re-trigger once label is set.

---

### Issue #8: Critic Retry Returns Worst Synthesis
**Fix location:** `agents/orchestrator.py`

**⚠ RISK:** On max retries exceeded, returning the last synthesis attempt may be worse than attempt #1.

**✔ FIX:** Store all `synthesis_attempts` as list of `(result, critic_score)` tuples. On exhaustion: `max(attempts, key=lambda x: x[1])`. Attach `critic_flag='max_retries_exceeded'` to response. Never return the last.

---

### Issue #9: Flask SSE Blocking / Orphaned Streams
**Fix location:** `web/app.py`

**⚠ RISK 1:** 8 chained LLM calls take 15–25 seconds. Default Flask WSGI blocks the thread.

**✔ FIX 1:** `stream_with_context()`. Each agent completion yields an SSE event. `X-Accel-Buffering: no` on all streaming routes. `threaded=True` in development.

**⚠ RISK 2:** User navigates away — server-side generator keeps running, burning API calls.

**✔ FIX 2:** Generator wrapped in `try/finally`. `finally` block sets cancellation flag. All agent calls check flag before executing. Session switch sends `closeSSE` event via JS before loading new session.

---

### Issue #10: History Compression Race Condition
**Fix location:** `session/history_store.py`

**⚠ RISK:** `compress_history()` runs in daemon thread. User sends new message before compression completes → two threads write SQLite simultaneously.

**✔ FIX:** `threading.Lock()` around all SQLite history writes. Additionally, enable WAL mode: `PRAGMA journal_mode=WAL` at SQLite connection setup. Compression fires after response is sent — never during response generation.

---

### Issue #11: Orchestrator Has No Global Timeout
**Fix location:** `agents/orchestrator.py`

**⚠ RISK:** Synthesis loop (2 passes) + critic retry loop (2 retries) = up to 4 synthesis + 4 critic calls. No bound on total execution time. SSE stream hangs indefinitely if Gemini is slow or retrying.

**✔ FIX:** Wrap LangGraph execution in `threading.Timer(60, abort_fn)`. On timeout: set abort flag, collect `synthesis_attempts` gathered so far, return best-scored attempt with flag `timeout_exceeded=True`. User sees a partial answer, not a hung spinner.

---

### Issue #12: Source Comparator "Same Story" Was Undefined
**Fix location:** `pipeline/source_comparator.py`

**⚠ RISK (v1.0 gap):** Original architecture never defined how "same story" is detected. Loose detection = comparator fires for every query (expensive). Tight detection = never fires.

**✔ FIX:** Extract top-5 named entities from each article title using a regex + stopword filter (no API call). If 2+ articles share ≥3 of the same entities AND were published within 24h → same story, fire comparator. Cheap, deterministic, no extra model call.

---

### Issue #13: No Query Cache (v1.0 gap)
**Fix location:** `infrastructure/query_cache.py`

**⚠ RISK:** Every test query reruns the full pipeline. During development, this burns the 2 RPM Pro quota within minutes. Same user asking a follow-up rephrasing triggers full synthesis again.

**✔ FIX:** `QueryCache`: `dict[str, (response, timestamp)]`. Key = `sha256(session_id + normalized_query)`. TTL = 60s. Check before LangGraph entry. Store after successful response. This is a Day 5 task — built at the same time as the orchestrator.

---

### Issue #14: Credibility Corroboration Inflated by Wire Republications
**Fix location:** `pipeline/credibility_scorer.py`

**⚠ RISK:** Two AP republications with slightly different text (evading hash dedup) both count as independent sources, inflating corroboration scores.

**✔ FIX:** Maintain a `WIRE_REPUBLISHER_DOMAINS` set (known wire-publishing aggregators). When counting corroboration, articles from this set count as 0.5 sources rather than 1.0. Full cluster detection deferred to v2.

---

## Project Structure

```
news-rag-system/
│
├── agents/
│   ├── orchestrator.py          ← LangGraph StateGraph, 11 nodes, 60s timeout
│   ├── intent_planner.py        ← combined intent + sub-query (1 Flash call)
│   ├── rag_agent.py             ← BM25 + ChromaDB + RRF + reranker
│   ├── web_search_agent.py      ← Tavily wrapper + dedup + date cap
│   ├── synthesis_agent.py       ← gemini-2.5-pro, iterative RAG (2 passes max)
│   ├── critic_agent.py          ← quality gate, circuit breaker, best-attempt return
│   └── formatter_agent.py       ← citations, badges, html.parser validation
│
├── pipeline/
│   ├── data_validator.py        ← length, paywall, language, wire-dedup hash
│   ├── crag_grader.py           ← batch grading, single Flash call
│   ├── credibility_scorer.py    ← 4-signal weighted score, URL cache, 0.5x wire weight
│   ├── bias_detector.py         ← 250-word gate, lean + confidence
│   ├── source_comparator.py     ← entity keyword trigger, 800-word truncation
│   └── hallucination_checker.py ← grounding check, flags ungrounded claims
│
├── retrieval/
│   ├── hybrid_retriever.py      ← BM25 + semantic + RRF, BM25 cache per session
│   └── reranker.py              ← ms-marco-MiniLM-L-6-v2, top_k=10 default
│
├── session/
│   ├── session_manager.py       ← create/load/delete, 300-chunk cap, ingested_at eviction
│   ├── history_store.py         ← SQLite, threading.Lock, WAL mode, async compression
│   └── topic_labeler.py         ← fires at >100 words, never re-triggers
│
├── infrastructure/
│   ├── google_client.py         ← generate(prompt, model, system), embed(text), tenacity
│   └── query_cache.py           ← sha256 key, 60s TTL, in-memory dict
│
├── data/
│   └── source_credibility_db.json  ← Tier 1–4 domains (50+ entries)
│
├── vectorstore/
│   └── chroma_store/            ← persistent, one collection per session
│
├── web/
│   ├── app.py                   ← Flask routes, SSE stream_with_context, threaded=True
│   ├── templates/
│   │   ├── index.html           ← Jinja2, EventSource JS, session sidebar
│   │   └── components/
│   │       ├── credibility_badge.html  ← green/amber/red, tooltip, no word "fake"
│   │       ├── bias_indicator.html     ← spectrum bar, confidence %, grey for insufficient
│   │       └── source_panel.html       ← agreed/disputed facts, framing differences
│   └── static/
│       ├── css/style.css
│       └── js/sse.js            ← EventSource handler, session switch, stream cancel
│
├── tests/
│   ├── fixtures/
│   │   └── cached_tavily_response.json  ← saved on Day 1, used in all unit tests
│   ├── smoke_p0.py
│   ├── smoke_p1.py
│   ├── smoke_p2.py
│   ├── smoke_p3.py
│   └── smoke_p4.py
│
├── config.py                    ← FAST_MODEL, SYNTH_MODEL, EMBED_MODEL, all thresholds
├── requirements.txt             ← pinned after smoke_p0 passes
├── .env.example
└── main.py
```

---

## Confirmed Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| LLM (fast) | gemini-2.0-flash | google-genai SDK | All agents except synthesis |
| LLM (reasoning) | gemini-2.5-pro | google-genai SDK | Synthesis only |
| Embeddings | text-embedding-004 | google-genai SDK | 768-dim, all vector embeds |
| Reranker | ms-marco-MiniLM-L-6-v2 | sentence-transformers | Cross-encoder, top_k=10, local CPU |
| Vector DB | ChromaDB | >=0.5, local persistent | One collection per session |
| Orchestration | LangGraph | >=0.1 | StateGraph, conditional edges, 60s timeout |
| Web Search | Tavily | tavily-python | Live news retrieval |
| BM25 | rank-bm25 | >=0.2.2 | Keyword search, per-session cached index |
| Backend | Flask | >=3.0 | Routes, SSE streaming |
| Frontend | Jinja2 + vanilla JS | Flask built-in | Templates + EventSource |
| Session DB | SQLite | stdlib | History + session mapping, WAL mode |
| Retry Logic | tenacity | >=8.4 | Backoff on all API calls |
| Testing | pytest | >=8.2 | Smoke tests, unit tests |

### requirements.txt

```
google-genai>=1.0
langchain>=0.2
langgraph>=0.1
chromadb>=0.5
tavily-python>=0.3
rank-bm25>=0.2.2
sentence-transformers>=2.7
flask>=3.0
flask-cors>=4.0
tenacity>=8.4
langdetect>=1.0.9
pytest>=8.2
python-dotenv>=1.0
```

> **Pin exact versions after first successful `smoke_p0.py` run. Replace `>=` with `==X.Y.Z` on every line. Never use `>=` in production.**

---

## Master Phase Overview

| Phase | Title | Key Deliverable | Target Days |
|-------|-------|----------------|-------------|
| 0 | Environment & Tooling | Google API live, ChromaDB running, keys verified | 1 |
| 1 | Core Pipeline | Orchestrator wired; sessions isolated; hybrid retrieval working | 2–5 |
| 2 | Intelligence Pipeline | Credibility, bias, CRAG, hallucination check integrated | 6–7 |
| 3 | Flask UI | SSE streaming chat, session sidebar, citation rendering | 8 |
| 4 | Hardening & Edge Cases | Error handling, rate-limit resilience, benchmarked | 9–10 |

**Total: 10 days.**

---

## PHASE 0 — Environment & Tooling
*All APIs verified · Keys tested · ChromaDB operational · No pipeline work*

---

### P0.1 — Repository & Project Structure

Create the canonical layout. Every subsequent task drops files into this structure.

- **Inputs:** GitHub account, Python 3.10+
- **Outputs:** Repo with full folder structure as above. `.env` added to `.gitignore` as the very first operation — before any other file is created or committed. `.env.example` committed with placeholder values.
- **Tools:** git, mkdir, python venv

> **⚠ RISK:** One accidental `.env` push and you rotate your API key.
>
> **✔ FIX:** `echo ".env" >> .gitignore` is the first command run in the repo. Verify with `git status` that `.env` does not appear before any commit.

---

### P0.2 — Google AI Studio + Tavily Connectivity

Verify both API keys work. Save Tavily fixture for all future tests.

- **Inputs:** `.env` with `GOOGLE_API_KEY`, `TAVILY_API_KEY`
- **Outputs:** `google_client.py` with `generate(prompt, model, system)` and `embed(text)` methods. Both methods wrapped in tenacity (min=2s, max=30s, max_attempts=4). `connectivity_test.py` prints `[OK]` or `[FAIL]` for each service. Tavily: fetch exactly one live query (`India news today`), save full response to `tests/fixtures/cached_tavily_response.json`. **Never make another live Tavily call in any test — always use the fixture.**
- **Tools:** google-genai, tavily-python, python-dotenv, tenacity

> **⚠ RISK:** `gemini-2.5-pro` is slow on first token (5–10s cold start). This is normal — do not mistake it for a connectivity failure.
>
> **✔ FIX:** Add a 30s timeout to the Pro connectivity test. Log the time-to-first-token. This is your baseline latency for synthesis planning.

---

### P0.3 — Model Smoke Test — Both Models + Embeddings

Call each model with a minimal prompt. Verify embeddings return the correct dimension.

- **Inputs:** `google_client.py` from P0.2
- **Outputs:** `smoke_p0_models.py` confirming: `gemini-2.0-flash ✓`, `gemini-2.5-pro ✓`, `text-embedding-004 ✓ (768-dim)`. Document `EMBED_DIM = 768` in `config.py`. Verify Pro raw output contains no leaked thinking tokens.
- **Tools:** google-genai SDK

> **⚠ RISK:** `text-embedding-004` has a 2048-token input limit. News articles chunked to 500–800 tokens are safe, but verify chunk size never exceeds 1800 tokens to leave headroom.
>
> **✔ FIX:** Add `assert len(chunk.split()) < 1500` in the embedder. Log a warning and truncate if exceeded.

---

### P0.4 — ChromaDB Naming & Session Manager Bootstrap

Test `_safe_collection_name()` on 20 edge cases. Verify SQLite schema.

- **Inputs:** chromadb, sqlite3
- **Outputs:** `_safe_collection_name()` passing 20 edge cases: UUID with hyphens, single-char string, 100-char string, all-hyphen string, string starting with hyphen, empty string. SQLite sessions table: `id, collection_name, topic_label, created_at, message_count`. Test session create → lookup → delete cycle. WAL mode enabled: `PRAGMA journal_mode=WAL`.
- **Tools:** chromadb, sqlite3, hashlib

> **⚠ RISK:** ChromaDB naming failures are silent in some versions — no exception, just wrong collection created.
>
> **✔ FIX:** After every `create_collection()`, immediately `get_collection()` with the same name and assert it returns. If it fails, the naming function is broken — fix before proceeding.

---

### 🧪 SMOKE TEST — Phase 0

Run: `python tests/smoke_p0.py`

1. `google_client.generate()` returns completion from Flash. Latency logged.
2. `google_client.generate()` returns completion from Pro. Time-to-first-token logged.
3. `google_client.embed()` returns vector of exactly 768 dimensions.
4. Tavily fixture exists at `tests/fixtures/cached_tavily_response.json`.
5. ChromaDB creates collection `stest001`, retrieves it by name, deletes it.
6. `_safe_collection_name()` passes all 20 edge cases.
7. SQLite sessions table created with correct schema, WAL mode confirmed.

**Fix every failure before Phase 1. All 7 must pass.**

---

## PHASE 1 — Core Pipeline
*Session management · Hybrid RAG · All agents · Orchestrator wired*

Build and test every component in isolation. Each agent has one public method and returns a typed dataclass. Wire the LangGraph orchestrator only after all agents pass individually. Do not build agents that depend on each other — the orchestrator is the only place coupling is allowed.

---

### P1.1 — Data Validator

Every Tavily result passes through this before touching ChromaDB.

- **Inputs:** None — pure Python utility
- **Outputs:** `pipeline/data_validator.py`: `DataValidator.validate(article)` → `(bool, dict)`. Checks: `has_url`, `content_length > 200`, `not_paywalled`, `not_empty_scrape`, `is_wire_duplicate` (MD5 of first 300 chars, lowercased), `content_length < 50000`, `published_date <= today`. Batch method: `validate_batch(articles, seen_hashes=None)` — `seen_hashes` is caller-controlled, per-batch scope. Never raises exception.
- **Tools:** hashlib, langdetect

> **⚠ RISK:** Global `seen_hashes` would flag legitimate re-coverage of ongoing stories.
>
> **✔ FIX:** Pass `seen_hashes` as a parameter. Caller creates a new `set()` for each Tavily batch. Per-batch dedup is correct; global dedup is wrong.

---

### P1.2 — Session Manager — Full Implementation

- **Inputs:** P0.4 ChromaDB + SQLite verified
- **Outputs:** `session/session_manager.py`: `create_session()` → `session_id`, `get_collection(session_id)` → ChromaDB collection, `add_chunks(session_id, chunks)` with 300-cap eviction by `ingested_at`, `delete_session(session_id)`, `validate_all_collections()` at startup. `session/history_store.py`: `append_message()`, `get_history()`, `compress_history()` (fires when `msg_count > 10`, summarises oldest 7 into 1 summary block, runs in daemon thread, guarded by `threading.Lock()`).
- **Tools:** chromadb, sqlite3, threading

> **⚠ RISK:** Compression fires during active response cycle at exactly message 11 — adds a Flash call to an already-heavy chain.
>
> **✔ FIX:** `compress_history()` only ever called via `threading.Thread(target=compress, daemon=True).start()` after the SSE stream closes. Never inline. `threading.Lock()` prevents race with concurrent appends.

---

### P1.3 — Hybrid Retriever + Reranker

- **Inputs:** P1.2 session manager, google_client embed, chromadb
- **Outputs:** `retrieval/hybrid_retriever.py`: `retrieve(query, session_id, top_k=10)` → `list[Chunk]`. BM25 index cached per `session_id` in a module-level dict. Cache invalidated when new chunks added to collection. Index built lazily on first query. `retrieval/reranker.py`: `rerank(query, chunks, top_n=5)` → `list[Chunk]`. Test: pre-populate session with 10 test articles. Verify reranked top-5 more relevant than semantic-only top-5.
- **Tools:** rank-bm25, chromadb, sentence-transformers

> **⚠ RISK:** BM25 rebuild for a 300-chunk session on every query is O(n) over all tokens.
>
> **✔ FIX:** Cache key = `(session_id, collection.count())`. When count changes, rebuild. When count is the same, return cached index. Add a timing log for rebuild — if >500ms, log a warning.

---

### P1.4 — Web Search Agent

- **Inputs:** P0.4 Tavily fixture, P1.1 DataValidator
- **Outputs:** `agents/web_search_agent.py`: `WebSearchAgent.search(query, sub_queries)` → `list[Article]`. `Article` dataclass: `url, outlet, published_date, ingested_at, content, credibility_score (placeholder 0), metadata_warning`. Articles deduplicated, date-capped, all passing DataValidator. In tests: always use `cached_tavily_response.json` fixture — never live Tavily.
- **Tools:** tavily-python, DataValidator

---

### P1.5 — CRAG Grader

- **Inputs:** P1.3 hybrid retriever, P1.4 web search agent
- **Outputs:** `pipeline/crag_grader.py`: `grade_documents(query, docs)` → `GradeResult`. `GradeResult`: `{grades: [{doc_id, grade}], routing_decision: rag_only|rag_plus_tavily|tavily_only}`. If `collection.count() < 5`: return `tavily_only` immediately, zero API calls. All documents graded in one Flash call: `"Grade relevance of each document. Return JSON array [{id, grade}]."`.
- **Tools:** gemini-2.0-flash via google_client

> **⚠ RISK:** CRAG grading 10 docs in 10 separate calls burns 10 RPM.
>
> **✔ FIX:** Batch prompt. One call regardless of document count. Verify with a test that 10 docs → exactly 1 API call.

---

### P1.6 — Intent + Query Planner

- **Inputs:** google_client, session history
- **Outputs:** `agents/intent_planner.py`: `IntentPlanner.plan(query, history_summary)` → `PlanResult`. `PlanResult`: `{intent: chitchat|followup|new_research, sub_queries: list[str], search_needed: bool, entity_heavy: bool}`. `entity_heavy=True` boosts BM25 weight in hybrid retriever. `history_summary` = `history_store.get_compressed_summary(session_id)` → max 500 words.
- **Tools:** gemini-2.0-flash, JSON structured output

---

### P1.7 — Synthesis Agent

- **Inputs:** P1.3, P1.4, P1.5, gemini-2.5-pro
- **Outputs:** `agents/synthesis_agent.py`: `SynthesisAgent.synthesize(context, query, history)` → `SynthesisResult`. `SynthesisResult`: `{answer, citations: [{source_url, chunk_text}], critic_score: 0.0 (placeholder), reasoning_gaps: list[str]}`. Iterative retrieval: up to 2 passes. Pass only triggered if `len(gap) > 5 words`. No reasoning token stripping — Pro handles this natively.
- **Tools:** gemini-2.5-pro via google_client

> **⚠ RISK:** Pro is 2 RPM. Iterative RAG can use up to 3 Pro calls (initial + 2 passes).
>
> **✔ FIX:** QueryCache check before synthesis entry. If a near-identical query was answered recently, reuse. For new queries: first pass is free, second pass only if gap is substantive, third pass never happens (cap = 2 iterations total).

---

### P1.8 — Critic Agent + Circuit Breaker

- **Inputs:** SynthesisResult from P1.7
- **Outputs:** `agents/critic_agent.py`: `CriticAgent.evaluate(synthesis_result, query)` → `CriticVerdict`. `CriticVerdict`: `{passed: bool, score: float 0-1, issues: list[str]}`. Orchestrator stores `(synthesis_result, score)` per attempt. On `max_retries=2` exhaustion: return `max(attempts, key=lambda x: x[1])`. Minimum pass threshold `score > 0.4` — any synthesis above this passes regardless of `passed` flag.
- **Tools:** gemini-2.0-flash via google_client

---

### P1.9 — Formatter Agent

- **Inputs:** SynthesisResult, credibility_map, bias_map, comparison_result
- **Outputs:** `agents/formatter_agent.py`: `FormatterAgent.format(synthesis, credibility_map, bias_map, source_comparison)` → `FormattedResponse`. `FormattedResponse`: `{answer_html, citations_html, credibility_badges_html, bias_panel_html, source_comparison_html}`. Every HTML fragment validated through `html.parser`. On parse failure: return plain-text fallback, never malformed HTML.
- **Tools:** gemini-2.0-flash, Jinja2 or f-strings

---

### P1.10 — QueryCache

- **Inputs:** None — pure Python utility
- **Outputs:** `infrastructure/query_cache.py`: `QueryCache` class. `store(session_id, query, response)`, `get(session_id, query)` → response or None. Key = `sha256((session_id + query.strip().lower()).encode())`. TTL = 60 seconds. In-memory dict `{key: (response, stored_at)}`. Evict expired entries on each `get()` call.
- **Tools:** hashlib, time

---

### P1.11 — LangGraph Orchestrator — Wire All Nodes

- **Inputs:** All agents P1.1–P1.10, LangGraph
- **Outputs:** `agents/orchestrator.py`: LangGraph StateGraph with all 11 nodes. State: typed dataclass, serialisable primitives only, no HTML blobs. HTML generation only in Formatter node. Global 60s timeout via `threading.Timer`. Nodes: `session_load, intent_plan, parallel_dispatch, data_validate, crag_grade, credibility_score, bias_detect, source_compare, synthesize, reliability_check, critique, format`. Conditional edges: `intent=chitchat → direct_response (skip all agents)`, `intent=followup + hot_cache → rag_only path`, `crag=tavily_only → skip rag_agent`, `critic_passed=False AND retries<2 → back to synthesize`.
- **Tools:** langgraph

> **⚠ RISK:** LangGraph state dict grows on every node. HTML blobs passed back through graph on retries are expensive.
>
> **✔ FIX:** State contains only IDs and primitives. HTML only in the final Formatter node.

---

### 🧪 SMOKE TEST — Phase 1

Run: `python tests/smoke_p1.py`

1. Create session, add 10 test chunks, verify `collection.count() == 10`.
2. Hybrid retriever query returns 5 reranked chunks.
3. DataValidator rejects a paywall stub and a wire duplicate.
4. CRAG grader batches 5 docs into exactly 1 API call.
5. Full pipeline: message → orchestrator → response in < 30 seconds.
6. Critic circuit breaker: force 2 failed syntheses, verify best attempt returned, not last.
7. Formatter produces valid HTML (`html.parser` accepts all fragments).
8. Cache hit returns immediately with 0 API calls.
9. Chitchat query skips all agents and returns in < 3 seconds.

**All 9 checks must pass.**

---

## PHASE 2 — Intelligence Pipeline
*Credibility scoring · Bias detection · Source comparison · Hallucination check*

Build each component in isolation with its own test fixture. These are pure data transformers — they do not call Tavily or modify session state. Wire into orchestrator only after all four pass independently.

---

### P2.1 — Credibility Scorer — 4-Signal Pipeline

- **Inputs:** `data/source_credibility_db.json`
- **Outputs:** `pipeline/credibility_scorer.py`: `CredibilityScorer.score(article)` → `CredibilityScore`. `CredibilityScore`: `{total: int, signals: {source_tier: int, corroboration: int, content_quality: int, freshness: int}}`. URL-keyed SQLite cache (`credibility_cache` table) — never re-score same URL. Corroboration: extract 3 core claims, find same claims in other batch sources. Counts: 0 independent → 10, 1 → 30, 2–3 → 60, 4+ → 85. Wire-republisher domains weighted at 0.5x per occurrence. `source_credibility_db.json` should contain 50+ domains across Tier 1–4 — generate it with a single Flash call if building from scratch.
- **Tools:** gemini-2.0-flash (content quality + claim extraction), sqlite3

---

### P2.2 — Bias Detector

- **Inputs:** Article content post-DataValidator
- **Outputs:** `pipeline/bias_detector.py`: `BiasDetector.detect(article)` → `BiasResult`. `BiasResult`: `{lean: Far-Left|Left|Center-Left|Center|Center-Right|Right|Far-Right|insufficient_content, confidence: float, bias_types: list[str], loaded_language: list[str], missing_perspectives: str}`. Zero API calls if `word_count < 250`.
- **Tools:** gemini-2.0-flash, JSON structured output

> **⚠ RISK:** Bias classification is subjective. Presenting it as objective fact undermines trust.
>
> **✔ FIX:** Always display as `Detected lean: Center-Right (62% confidence)` — never as binary `biased`. Tooltip: `Automated classification. May not reflect full context.` Never display `Far-Left` or `Far-Right` as an accusation — just a detected signal.

---

### P2.3 — Source Comparator

- **Inputs:** 2+ validated articles passing same-story trigger
- **Outputs:** `pipeline/source_comparator.py`: `SourceComparator.compare(articles)` → `ComparisonResult`. `ComparisonResult`: `{agreed_facts, disputed_facts, unique_claims: dict[outlet → list], framing_differences}`. Same-story trigger: extract top-5 named entities from each title using regex + stopword filter (no API call). If ≥3 shared entities AND published within 24h → fire. First 800 words per article passed to comparator. Add note: `Comparison based on first 800 words of each article.`
- **Tools:** gemini-2.0-flash

---

### P2.4 — Hallucination / Reliable RAG Check

- **Inputs:** SynthesisResult + retrieved chunks used in synthesis
- **Outputs:** `pipeline/hallucination_checker.py`: `HallucinationChecker.check(synthesis, source_chunks)` → `HallucinationReport`. `HallucinationReport`: `{grounded_claims, ungrounded_claims, grounding_score: float}`. Ungrounded claims shown with ⚠ marker. `grounding_score` tooltip: `Measures whether claims are supported by retrieved sources, not absolute factual accuracy.`
- **Tools:** gemini-2.0-flash, JSON structured output

---

### P2.5 — Integrate Intelligence Pipeline into Orchestrator

- **Inputs:** All P2 components, Orchestrator from P1.11
- **Outputs:** Updated `orchestrator.py`. Credibility + bias run sequentially per article (not in parallel — parallel means simultaneous Flash calls, sequential spreads them over time). Source comparator fires on keyword trigger. State updated with `credibility_map, bias_map, comparison_result, hallucination_report`.
- **Tools:** langgraph

> **⚠ RISK:** Running credibility + bias on 5 articles simultaneously means 10 concurrent Flash calls, hitting 15 RPM limit within seconds.
>
> **✔ FIX:** Sequential per article. 5 articles × 2 calls = 10 calls spread over ~5–8 seconds at normal Flash speed. Acceptable.

---

### 🧪 SMOKE TEST — Phase 2

Run: `python tests/smoke_p2.py`

1. CredibilityScorer: known major outlet article scores > 70. Unknown blog scores < 35.
2. BiasDetector: 150-word article returns `insufficient_content` with zero API calls confirmed via mock.
3. SourceComparator: two articles about same event return at least 2 `agreed_facts`.
4. HallucinationChecker: synthesis with a fabricated claim returns that claim in `ungrounded_claims`.
5. Full pipeline with intelligence layer: end-to-end < 45 seconds.

**All 5 must pass.**

---

## PHASE 3 — Flask UI
*Session sidebar · Streaming responses · Citation rendering · Credibility badges*

Build only after Phase 1 + 2 smoke tests pass. No Stitch, no HTMX. Plain Flask templates with native browser EventSource. The streaming connection and the session sidebar are the two critical features — build them first, styling second.

---

### P3.1 — Flask App + SSE Streaming

- **Inputs:** Phase 1+2 pipeline complete
- **Outputs:** `web/app.py` with routes: `GET /` (chat UI), `POST /chat` (SSE stream), `POST /session/new`, `GET /sessions`, `DELETE /session/<id>`. SSE stream yields JSON events: `{type: status|partial|complete, agent: str, content: str}`. Header `X-Accel-Buffering: no` on all streaming routes. `threaded=True`. Generator in `try/finally` with cancellation flag.
- **Tools:** flask, flask-cors

> **⚠ RISK:** User navigates away — generator keeps running, burns Pro quota.
>
> **✔ FIX:** `try/finally` in generator. `finally` sets `cancelled = True`. Every agent call begins with `if cancelled: raise GeneratorExit`.

---

### P3.2 — Jinja2 Templates + EventSource JS

- **Inputs:** Flask routes from P3.1
- **Outputs:** `web/templates/index.html` with: chat window, session sidebar (rendered via `fetch('/sessions')` on page load), message input form. `web/static/js/sse.js`: `EventSource` opened on form submit, events appended to chat window, `source.close()` called on session switch before opening new connection. No external JS dependencies beyond what Flask provides.
- **Tools:** Jinja2, vanilla JavaScript

> **⚠ RISK:** Native `EventSource` is GET-only in some browsers. Flask `/chat` must use GET with query params, or use a POST-then-GET pattern (POST creates a job_id, GET streams results by job_id).
>
> **✔ FIX:** Use GET with query params: `GET /chat?q=<query>&session_id=<id>`. Query and session_id are URL-encoded. This is the simplest pattern and works in all browsers without polyfills.

---

### P3.3 — Credibility Badges + Bias Indicators

- **Inputs:** FormattedResponse HTML fragments from P1.9
- **Outputs:** Jinja2 macros in `templates/components/`. Credibility badge: green (>70), amber (40–70), red (<40). Score number + source name. Hover tooltip: signal breakdown. Grey badge for `insufficient_content`. Bias indicator: lean label + confidence %. Tooltip: `Automated classification. May not reflect full context.` Never use the word `fake` anywhere.
- **Tools:** Jinja2, CSS

---

### P3.4 — Session Sidebar + Topic Labels

- **Inputs:** session_manager, topic_labeler
- **Outputs:** Session list updated via `fetch('/sessions')`. Each session shows `topic_label`, `created_at` (relative: `2 hours ago`), `message_count`. Clicking a session: `source.close()` on active EventSource → load history → render in chat window. `POST /session/new` → creates session → updates sidebar.
- **Tools:** Flask, Jinja2, vanilla JS

---

### 🧪 SMOKE TEST — Phase 3

Run: `python tests/smoke_p3.py`

1. `GET /` returns 200 and valid HTML.
2. `POST /session/new` creates a session and returns `session_id`.
3. `GET /chat?q=test&session_id=<id>` streams at least 3 SSE events.
4. Response HTML passes `html.parser` validation.
5. Session sidebar shows new session with label `New Research`.
6. After 2 messages totalling > 100 words, session label updates.
7. Session switch closes active EventSource cleanly — no orphaned stream.

**All 7 must pass.**

---

## PHASE 4 — Hardening & Edge Cases
*Error handling · Rate-limit resilience · Load tested*

---

### P4.1 — Google API Failure Simulation

- **Outputs:** `tests/test_google_resilience.py`: (1) Mock 429 → verify tenacity retries with backoff, does not crash, returns graceful error after `max_attempts`. (2) Mock timeout → verify 60s hard abort returns best synthesis attempt with `timeout_exceeded=True`. (3) Mock truncated JSON response → verify `json_parser` fallback returns safe default.
- **Tools:** pytest, unittest.mock

---

### P4.2 — ChromaDB Edge Case Tests

- **Outputs:** `tests/test_session_edge_cases.py`: (1) Collection at 300-chunk cap → new add evicts 50 oldest by `ingested_at`, count stays ≤ 300. (2) Lookup nonexistent `session_id` → `SessionNotFoundError` raised cleanly. (3) `_safe_collection_name()` on 20 pathological inputs — all produce valid names. (4) `validate_all_collections()` with 1 orphaned session → logs warning, does not crash.
- **Tools:** pytest, chromadb

---

### P4.3 — Tavily Edge Cases

- **Outputs:** `tests/test_tavily_edge_cases.py`: (1) Tavily returns 0 results → pipeline routes to knowledge-cutoff response, not crash. (2) Article with future date → date capped at today, `metadata_warning` set, `ingested_at` used for eviction. (3) All 5 articles from same wire source → all but first deduplicated, pipeline continues with 1 article.
- **Tools:** pytest, unittest.mock

---

### P4.4 — UI Edge Cases

- **Outputs:** (1) SSE stream drops mid-response → UI shows `Connection lost`, does not hang. (2) Session switch during active stream → stream cancelled, new session loads cleanly. (3) Empty message → rejected client-side before EventSource is opened. (4) Malformed HTML from formatter → plain-text fallback renders.
- **Tools:** pytest, manual checklist

---

### P4.5 — Response Time Benchmarking

- **Outputs:** Benchmark script: 10 varied queries (follow-up, new research, chitchat). Record total time and per-agent time. Targets: chitchat < 3s, follow-up < 15s, new research < 35s. Any path exceeding target → profile with `cProfile`.

> **⚠ RISK:** Cross-encoder reranker on CPU adds 200ms–1s. Pro synthesis adds 5–10s cold start.
>
> **✔ FIX:** Reranker is already defaulted to top_k=10 (not 20). Pro synthesis is only called once per new_research query — cache and iterative-pass cap keep it bounded. If synthesis > 20s, check if Pro is returning a full thinking trace — confirm it's stripped at API level.

---

### 🧪 SMOKE TEST — Phase 4

Run: `python tests/smoke_p4.py`

1. All resilience, edge case, and benchmark tests pass.
2. No test produces an unhandled exception.
3. Response time targets met on 8 of 10 test queries.
4. 60s timeout fires correctly on a simulated slow synthesis.
5. ChromaDB cap enforced on 300-chunk boundary.

**This is the final smoke test. The system is production-ready when this passes.**

---

## Master Timeline

| Day | Phase | Primary Tasks | End-of-Day Checkpoint |
|-----|-------|---------------|-----------------------|
| 1 | P0 | Repo, .env, google_client, Tavily fixture, ChromaDB naming | smoke_p0.py: all models OK, 768-dim embedding, naming passes 20 cases |
| 2 | P1 | DataValidator, SessionManager (with Lock + WAL), HistoryStore | Session create/load/delete. DataValidator rejects wire duplicate. |
| 3 | P1 | HybridRetriever (BM25 cached), Reranker (top_k=10), WebSearchAgent | Reranked top-5 more relevant than semantic top-5. |
| 4 | P1 | CRAG grader (batched), IntentPlanner, SynthesisAgent, CriticAgent | CRAG: 5 docs → 1 API call. Critic circuit breaker returns best attempt. |
| 5 | P1 | FormatterAgent, QueryCache, LangGraph orchestrator | smoke_p1.py: all 9 checks pass. Cache hit = 0 API calls. |
| 6 | P2 | CredibilityScorer (4 signals, URL cache), BiasDetector | Major outlet > 70. Short article returns insufficient_content, 0 calls. |
| 7 | P2 | SourceComparator (keyword trigger), HallucinationChecker, integrate P2 | smoke_p2.py: full pass. End-to-end < 45s. |
| 8 | P3 | Flask app, SSE (GET pattern), Jinja2 templates, session sidebar, badges | smoke_p3.py: all 7 checks pass. Session switch closes stream cleanly. |
| 9 | P4 | Google API resilience, ChromaDB edge cases, Tavily edge cases | All resilience tests pass. No unhandled exceptions. |
| 10 | P4 | UI edge cases, benchmarking, pin requirements.txt, final checklist | smoke_p4.py: full pass. Targets met. Checklist complete. |

### Critical Path

Any delay here propagates directly to the deadline:

- **P0.2 (google_client)** must be built before any agent work — every agent calls it
- **P0.4 (ChromaDB naming)** must be validated before P1.2 (SessionManager)
- **P1.3 (Hybrid Retriever)** must pass before P1.5 (CRAG) — CRAG decides if retrieval is good enough
- **P1.10 (QueryCache)** must exist before P1.11 (Orchestrator) — Pro rate limit makes cache mandatory
- **P1.11 (Orchestrator)** is last in Phase 1 — wired only after all agents individually pass
- **smoke_p1.py** must pass before Phase 2 — non-negotiable gate
- **P2.5 (integrate intelligence)** must complete before Phase 3
- **P3.1 (Flask SSE)** must work before P3.2 — cannot build UI against a broken stream
- **smoke_p4.py** is the final production readiness gate

---

## Pre-Submission Checklist

### Infrastructure
- [ ] `smoke_p0.py` through `smoke_p4.py` all pass without modification
- [ ] `config.py` defines `FAST_MODEL`, `SYNTH_MODEL`, `EMBED_MODEL` — no string literals in agent code
- [ ] No API keys in any source file — all from `.env` via python-dotenv
- [ ] `requirements.txt` has pinned exact versions (`==X.Y.Z`)
- [ ] tenacity backoff wraps every google_client call — no bare API calls anywhere
- [ ] `GOOGLE_API_KEY` and `TAVILY_API_KEY` are the only secrets in `.env`

### Session & Storage
- [ ] `_safe_collection_name()` passes 20 edge cases
- [ ] SQLite `session_id → collection_name` mapping — never reconstructed from session_id
- [ ] `validate_all_collections()` runs at startup without crashing
- [ ] 300-chunk cap enforced — `collection.count()` never exceeds 300
- [ ] Eviction uses `ingested_at` metadata, not `published_date`
- [ ] History compression fires asynchronously with `threading.Lock()`
- [ ] WAL mode enabled on SQLite connection: `PRAGMA journal_mode=WAL`
- [ ] Topic labels update only when `word_count > 100` AND label is still `New Research`

### Pipeline Correctness
- [ ] Wire duplicate detection uses per-batch `seen_hashes` set — not global
- [ ] CRAG grader batches all docs into single API call
- [ ] Bias detector skips articles < 250 words with zero API calls
- [ ] Critic circuit breaker returns BEST attempt (by score), not last
- [ ] Synthesis iterative pass capped at 2 iterations
- [ ] Synthesis iterative pass only fires if gap > 5 words
- [ ] Hallucination report shows ungrounded claims — never hidden
- [ ] Source comparator trigger: ≥3 shared entities + within 24h (no API call for trigger)
- [ ] Source comparator truncates to first 800 words per article
- [ ] QueryCache checked before LangGraph entry — cache hit = 0 API calls
- [ ] Global 60s orchestrator timeout — returns best attempt with `timeout_exceeded=True`
- [ ] Pro model used only for synthesis — every other node uses Flash

### UI & Streaming
- [ ] SSE uses GET pattern: `GET /chat?q=<query>&session_id=<id>`
- [ ] `X-Accel-Buffering: no` header on all streaming routes
- [ ] Generator wrapped in `try/finally` with cancellation flag
- [ ] Session switch calls `source.close()` before opening new EventSource
- [ ] Empty message rejected client-side — no API calls
- [ ] Malformed formatter HTML falls back to plain text
- [ ] Credibility badges: green (>70), amber (40–70), red (<40), grey (insufficient_content)
- [ ] Bias display: lean + confidence % — never binary label, never word "fake"
- [ ] Grounding score tooltip: `Measures source-grounding, not absolute accuracy`
- [ ] Credibility badge tooltip: `Limited corroboration. May be breaking news or lesser-known outlet`

---

*Real-Time News RAG — Multi-Agent Research & Credibility System | Architecture & Build Roadmap v2.0 | Stack: Google AI Studio · Flask · ChromaDB · LangGraph · Tavily*

# 🔬 News Analyzer

**AI-powered news research platform with multi-agent intelligence pipeline, real-time source verification, and Gemini-style live execution trace.**

News Analyzer uses a LangGraph-orchestrated multi-agent pipeline powered by Google Gemini to research any news topic. It searches the web, retrieves from a local knowledge base, scores source credibility, detects bias, checks for hallucinations, and synthesizes a verified, citation-backed answer — all while streaming each step to the browser in real-time.

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version |
|------------|---------|
| Python | 3.10+ |
| Google Gemini API Key | [Get one here](https://aistudio.google.com/app/apikey) |
| Tavily API Key | [Get one here](https://tavily.com) |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/harshdeep220/news-analyzer.git
cd news-analyzer

# 2. Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# Edit .env with your API keys:
#   GOOGLE_API_KEY=your_google_api_key_here
#   TAVILY_API_KEY=your_tavily_api_key_here

# 5. Run database migrations
python web/manage.py migrate

# 6. Start the server
python web/manage.py runserver 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### CLI Mode (No Browser)

```bash
python cli.py
```

This starts an interactive terminal session with the same pipeline, useful for debugging and testing.

---

## 🏗️ Architecture

```
news-analyzer/
├── agents/                    # LLM-powered agent modules
│   ├── orchestrator.py        # LangGraph StateGraph — wires everything together
│   ├── intent_planner.py      # Query classification and sub-query generation
│   ├── web_search_agent.py    # Tavily web search with result parsing
│   ├── synthesis_agent.py     # Multi-pass answer synthesis with gap retrieval
│   ├── critic_agent.py        # Quality gate with circuit breaker (max 2 retries)
│   └── formatter_agent.py     # HTML/JSON output formatting with citations
│
├── pipeline/                  # Intelligence analysis components
│   ├── data_validator.py      # Source quality filtering (length, language, dedup)
│   ├── crag_grader.py         # Corrective RAG — relevance grading & routing
│   ├── credibility_scorer.py  # 4-signal credibility scoring (0–100)
│   ├── bias_detector.py       # Political/ideological bias classification
│   ├── source_comparator.py   # Cross-source fact comparison & consensus
│   └── hallucination_checker.py # Post-synthesis claim grounding verification
│
├── retrieval/                 # Document retrieval
│   ├── hybrid_retriever.py    # Semantic (ChromaDB) + BM25 hybrid search
│   └── reranker.py            # Cross-encoder reranking (ms-marco-MiniLM)
│
├── infrastructure/            # Shared infrastructure
│   ├── google_client.py       # Gemini API client with retry + rate limiting
│   └── query_cache.py         # In-memory TTL cache (60s) for repeat queries
│
├── web/                       # Django web application
│   ├── manage.py              # Django entry point
│   ├── newsrag/               # Django project settings
│   │   ├── settings.py        # Database, static files, middleware config
│   │   ├── urls.py            # Root URL routing
│   │   └── wsgi.py            # WSGI application
│   ├── chat/                  # Main Django app
│   │   ├── models.py          # Session, Message, CredibilityCache models
│   │   ├── services.py        # SessionManager, HistoryStore, TopicLabeler
│   │   ├── views.py           # SSE streaming, session CRUD, page views
│   │   ├── urls.py            # API routing
│   │   └── admin.py           # Django admin registration
│   ├── templates/             # HTML templates
│   │   ├── base.html          # Base layout (fonts, meta)
│   │   └── index.html         # Chat UI with inline trace panel
│   └── static/                # Frontend assets
│       ├── css/style.css      # Gemini-inspired dark theme
│       └── js/app.js          # SSE client, trace renderer, session management
│
├── config.py                  # Central configuration (models, thresholds, weights)
├── cli.py                     # Interactive CLI for terminal testing
├── data/                      # SQLite databases (auto-created)
└── vectorstore/               # ChromaDB persistent storage
```

---

## 🔄 Pipeline Flow

The orchestrator uses **LangGraph's `StateGraph`** to execute 13 nodes in a directed acyclic graph:

```
┌──────────────┐
│ Session Load │ ── Load history, ChromaDB collection
└──────┬───────┘
       ▼
┌──────────────┐     ┌─────────────┐
│ Intent Plan  │────▶│ Direct End  │  (chat/greeting → skip pipeline)
└──────┬───────┘     └─────────────┘
       ▼
┌──────────────────────┐
│ Parallel Dispatch    │ ── RAG retrieval + Tavily web search (concurrent)
└──────┬───────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Data Validate│────▶│ CRAG Grade   │────▶│ Credibility  │
│ (filter junk)│     │ (relevance)  │     │ Score (0-100)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Source       │◀────│ Bias Detect  │◀────│              │
│ Compare      │     │ (lean + conf)│     │              │
└──────┬───────┘     └──────────────┘     └──────────────┘
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Synthesize   │────▶│ Hallucination│────▶│ Critic       │
│ (Gemini Pro) │     │ Check        │     │ (pass/fail)  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                          ┌───────┴───────┐
                                          │  Retry?       │
                                          │  (max 2x)     │
                                          └───────┬───────┘
                                                  ▼
                                          ┌──────────────┐
                                          │ Format Output│ ── HTML + citations
                                          └──────────────┘
```

---

## 🧩 Components

### Agents (LLM-powered)

| Agent | Model | Purpose |
|-------|-------|---------|
| **Intent Planner** | Gemini 2.5 Flash | Classifies query intent (`new_research`, `follow_up`, `chat`), extracts entities, generates sub-queries |
| **Web Search Agent** | Tavily API | Executes up to 4 parallel web searches, parses and normalizes results |
| **Synthesis Agent** | Gemini 2.5 Pro | Multi-pass answer generation with iterative gap retrieval (max 2 passes) |
| **Critic Agent** | Gemini 2.5 Flash | Quality gate — scores on accuracy, completeness, bias. Circuit breaker with max 2 retries |
| **Formatter Agent** | Gemini 2.5 Flash | Converts synthesis to structured HTML with inline citations |

### Intelligence Pipeline

| Component | Purpose |
|-----------|---------|
| **Data Validator** | Filters articles by length (200–50000 chars), language (English), and wire-story deduplication |
| **CRAG Grader** | Corrective RAG — grades each chunk for relevance, routes to `web_only`/`augmented`/`tavily_only` |
| **Credibility Scorer** | 4-signal weighted score: source tier (35%), corroboration (30%), content quality (20%), freshness (15%) |
| **Bias Detector** | Classifies political lean (`left`, `center-left`, `center`, `center-right`, `right`) with confidence |
| **Source Comparator** | Detects same-story coverage, identifies consensus claims and contradictions across outlets |
| **Hallucination Checker** | Post-synthesis grounding — verifies each claim against retrieved sources, flags ungrounded statements |

### Retrieval Layer

| Component | Purpose |
|-----------|---------|
| **Hybrid Retriever** | Combines semantic search (ChromaDB + Gemini embeddings) with BM25 keyword search |
| **Reranker** | Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2` for precision |

---

## 🎨 Web Interface

The Django-based UI features a **Gemini-inspired dark theme** with:

- **Real-time Agent Trace** — Inline collapsible "Analyzing..." block shows each pipeline step as it executes (spinner → checkmark), with elapsed time
- **Session Management** — Create, switch, and delete research sessions via the sidebar
- **Credibility Badges** — Color-coded scores (🟢 >70, 🟡 40–70, 🔴 <40) for each source
- **Bias Indicators** — Political lean classification with confidence percentage
- **Hallucination Warnings** — Ungrounded claims flagged with grounding score

### Live Agent Trace

As the pipeline runs, the UI displays step-by-step progress:

```
▼ Analyzing... (12.3s)
  ✓ Loading session         0.2s
  ✓ Analyzing intent        1.8s
  ✓ Searching the web       4.2s
  ✓ Validating sources      4.3s
  ✓ Grading relevance       5.1s
  ✓ Scoring credibility     5.4s
  ✓ Detecting bias          5.6s
  ⏳ Synthesizing answer...
```

This is implemented via **Server-Sent Events (SSE)** using Django's `StreamingHttpResponse` with a `queue.Queue` bridge to the background pipeline thread.

---

## 🔌 API Reference

All endpoints are served from the Django app at `http://localhost:8000`.

### Pages

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Main chat page |

### SSE Streaming

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/chat?q=...&session_id=...` | Streams pipeline events as SSE |

**SSE Event Types:**

```jsonc
// Status update (one per pipeline step)
{"type": "status", "agent": "intent_planner", "content": "Analyzing query...", "timestamp": "..."}

// Final result
{"type": "done", "result": {"answer": "...", "credibility": {...}, "bias": {...}, "hallucination": {...}}}

// Error
{"type": "error", "content": "Pipeline failed: ..."}
```

### Session CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/session/new` | Create new session → `{session_id, topic_label}` |
| `GET` | `/api/sessions` | List all sessions → `{sessions: [...]}` |
| `GET` | `/api/session/<id>/history` | Get message history → `{messages: [...]}` |
| `DELETE` | `/api/session/<id>/delete` | Delete a session → `{deleted: true}` |

---

## ⚙️ Configuration

All configuration lives in `config.py` — no magic strings elsewhere.

### Models

| Constant | Value | Usage |
|----------|-------|-------|
| `FAST_MODEL` | `gemini-2.5-flash` | All agents except synthesis |
| `SYNTH_MODEL` | `gemini-2.5-pro` | Synthesis + deep reasoning |
| `EMBED_MODEL` | `gemini-embedding-2` | All vector embeddings (768-dim) |

### Key Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ORCHESTRATOR_TIMEOUT` | 120s | Hard pipeline abort |
| `CRITIC_MAX_RETRIES` | 2 | Circuit breaker limit |
| `CRITIC_PASS_THRESHOLD` | 0.4 | Minimum quality score |
| `QUERY_CACHE_TTL` | 60s | Cache expiry |
| `CRAG_COLD_SESSION_THRESHOLD` | 5 | Docs needed before RAG-only mode |
| `SYNTHESIS_MAX_PASSES` | 2 | Iterative retrieval cap |

### Credibility Weights

| Signal | Weight |
|--------|--------|
| Source Tier | 35% |
| Corroboration | 30% |
| Content Quality | 20% |
| Freshness | 15% |

---

## 💾 Database

News Analyzer uses **Django ORM** with SQLite (at `data/newsrag.db`).

### Models

| Model | Fields | Purpose |
|-------|--------|---------|
| `Session` | `id (UUID)`, `topic_label`, `collection_name`, `message_count`, `created_at` | Research sessions |
| `Message` | `session (FK)`, `role`, `content`, `is_summary`, `created_at` | Chat history with compression |
| `CredibilityCache` | `url (PK)`, `total`, `signals (JSON)`, `scored_at` | Cached credibility scores |

### Services

The `chat/services.py` module provides three service classes that wrap the Django ORM:

- **`SessionManager`** — Create, list, delete sessions. Auto-generates ChromaDB collection names.
- **`HistoryStore`** — Append messages, retrieve history, async compression (summarizes when >10 messages).
- **`TopicLabeler`** — Auto-labels sessions after enough content using Gemini Flash.

---

## 🛠️ Development

### Adding a New Pipeline Node

1. Create your component in `pipeline/` or `agents/`
2. Add the node function in `agents/orchestrator.py`
3. Register with `graph.add_node("your_node", your_function)`
4. Wire edges with `graph.add_edge()`
5. Add event emission: `_emit_event_dict(state, "your_node", "status", "Description...")`
6. Add a step entry in `web/static/js/app.js` → `STEPS` array

### Running Tests

```bash
# Run the test suite
python -m pytest tests/ -v

# Run with debug logging
set DEBUG=true && python web/manage.py runserver 8000
```

### Project Commands

```bash
# Start web server
python web/manage.py runserver 8000

# Interactive CLI
python cli.py

# Django admin (create superuser first)
python web/manage.py createsuperuser
python web/manage.py runserver 8000
# Then visit http://localhost:8000/admin/

# Database migrations
python web/manage.py makemigrations chat
python web/manage.py migrate

# Visualize the pipeline graph
python graph_vislualize.py
```

---

## 🔒 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ | Google Gemini API key |
| `TAVILY_API_KEY` | ✅ | Tavily web search API key |

Create a `.env` file in the project root (use `.env.example` as template).

---

## 📜 License

This project is for educational and research purposes.
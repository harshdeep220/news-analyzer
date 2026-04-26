"""
Central configuration — Single source of truth for all model IDs, thresholds, and constants.
No model string or threshold value appears anywhere else in the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Keys ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ─── Model IDs ────────────────────────────────────────────────────────────────
FAST_MODEL = "gemini-2.5-flash"          # All agents except synthesis
SYNTH_MODEL = "gemini-2.5-pro"           # Synthesis + deep reasoning only
EMBED_MODEL = "gemini-embedding-2"       # All vector embeddings — 768-dim output (via MRL)

# ─── Embedding ────────────────────────────────────────────────────────────────
EMBED_DIM = 768                          # text-embedding-004 output dimension
EMBED_MAX_TOKENS = 1500                  # Safety cap (model limit is 8192, we keep conservative)
EMBED_OUTPUT_DIM = 768                   # MRL dimensionality — gemini-embedding-2 default is 3072

# ─── Session & Storage ────────────────────────────────────────────────────────
MAX_CHUNKS_PER_SESSION = 300             # ChromaDB collection cap per session
EVICTION_BATCH_SIZE = 50                 # Chunks evicted when cap reached
SQLITE_DB_PATH = os.path.join("data", "sessions.db")

# ─── Rate Limits & Caching ────────────────────────────────────────────────────
QUERY_CACHE_TTL = 60                     # Seconds before cache entry expires
RETRY_MIN_WAIT = 2                       # Tenacity exponential backoff min (seconds)
RETRY_MAX_WAIT = 30                      # Tenacity exponential backoff max (seconds)
RETRY_MAX_ATTEMPTS = 4                   # Tenacity max retry attempts

# ─── Orchestrator ─────────────────────────────────────────────────────────────
ORCHESTRATOR_TIMEOUT = 60                # Hard abort (seconds)
CRITIC_MAX_RETRIES = 2                   # Circuit breaker max retries
CRITIC_PASS_THRESHOLD = 0.4             # Minimum score to pass regardless of flag

# ─── Pipeline Thresholds ──────────────────────────────────────────────────────
HISTORY_COMPRESS_THRESHOLD = 10          # Messages before async compression fires
HISTORY_MAX_SUMMARY_WORDS = 500          # Max words in compressed history summary
TOPIC_LABEL_WORD_THRESHOLD = 100         # Total word count before topic labeling
BIAS_MIN_WORDS = 250                     # Below this → insufficient_content, 0 API calls
COMPARATOR_ENTITY_THRESHOLD = 3          # Shared entities required to trigger comparison
COMPARATOR_TIME_WINDOW_HOURS = 24        # Max hours between articles for same-story
COMPARATOR_MAX_WORDS = 800               # First N words per article sent to comparator
SYNTHESIS_MAX_PASSES = 2                 # Iterative retrieval cap
SYNTHESIS_GAP_MIN_WORDS = 5              # Gap must exceed this to justify another pass

# ─── Retrieval ────────────────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_K = 10                      # Candidates fed to reranker
RERANKER_TOP_N = 5                       # Results returned after reranking

# ─── Data Validation ─────────────────────────────────────────────────────────
MIN_CONTENT_LENGTH = 200                 # Minimum chars for valid article
MAX_CONTENT_LENGTH = 50000               # Maximum chars for valid article
WIRE_DEDUP_CHARS = 300                   # First N chars for MD5 wire-story dedup

# ─── Credibility Scoring Weights ──────────────────────────────────────────────
CRED_WEIGHT_SOURCE_TIER = 0.35
CRED_WEIGHT_CORROBORATION = 0.30
CRED_WEIGHT_CONTENT_QUALITY = 0.20
CRED_WEIGHT_FRESHNESS = 0.15

# ─── CRAG ─────────────────────────────────────────────────────────────────────
CRAG_COLD_SESSION_THRESHOLD = 5          # Below this doc count → tavily_only

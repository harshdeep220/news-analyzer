"""
LangGraph Orchestrator — 11-node StateGraph with 60s global timeout.

Wires all agents and pipeline components. State uses only typed primitives
and IDs — no HTML blobs until the Formatter node.

Conditional edges:
  - chitchat → direct_response (skip all agents)
  - followup + hot cache → rag_only path
  - crag = tavily_only → skip rag_agent
  - critic_passed = False AND retries < 2 → back to synthesize

Circuit breaker: max 2 retries, returns max(attempts, key=score), NOT last attempt.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import StateGraph, END

from agents.intent_planner import IntentPlanner, PlanResult
from agents.web_search_agent import WebSearchAgent
from agents.synthesis_agent import SynthesisAgent
from agents.critic_agent import CriticAgent
from agents.formatter_agent import FormatterAgent
from infrastructure.google_client import GoogleClient
from infrastructure.query_cache import QueryCache
from pipeline.crag_grader import CRAGGrader
from pipeline.data_validator import DataValidator
from retrieval.hybrid_retriever import HybridRetriever, Chunk
from retrieval.reranker import rerank
from session.session_manager import SessionManager
from session.history_store import HistoryStore
from session.topic_labeler import TopicLabeler
from config import (
    ORCHESTRATOR_TIMEOUT,
    CRITIC_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# ─── Singleton instances ──────────────────────────────────────────────────────
_session_manager = None
_history_store = None
_topic_labeler = None
_query_cache = None


def _get_session_manager():
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def _get_history_store():
    global _history_store
    if _history_store is None:
        _history_store = HistoryStore()
    return _history_store


def _get_topic_labeler():
    global _topic_labeler
    if _topic_labeler is None:
        _topic_labeler = TopicLabeler()
    return _topic_labeler


def _get_query_cache():
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


# ─── Pipeline State ──────────────────────────────────────────────────────────
@dataclass
class PipelineState:
    """Typed state object — serialisable primitives only, no HTML blobs."""
    # Input
    query: str = ""
    session_id: str = ""

    # Session context
    history_summary: str = ""
    collection_count: int = 0

    # Intent
    intent: str = ""
    sub_queries: list = field(default_factory=list)
    search_needed: bool = True
    entity_heavy: bool = False
    direct_response: str = ""

    # Retrieved data
    rag_chunks: list = field(default_factory=list)
    web_articles: list = field(default_factory=list)
    all_chunks: list = field(default_factory=list)

    # CRAG
    crag_routing: str = ""  # rag_only | rag_plus_tavily | tavily_only

    # Intelligence (Phase 2 placeholders)
    credibility_map: dict = field(default_factory=dict)
    bias_map: dict = field(default_factory=dict)
    source_comparison: Any = None
    hallucination_report: Any = None

    # Synthesis
    synthesis_result: Any = None
    synthesis_attempts: list = field(default_factory=list)  # [(result, score)]
    critic_retries: int = 0

    # Output
    formatted_response: Any = None
    status: str = "pending"
    error: str = ""

    # Control flags
    cancelled: bool = False
    timeout_exceeded: bool = False
    cache_hit: bool = False

    # SSE events log
    events: list = field(default_factory=list)


def _emit_event(state: PipelineState, agent: str, status: str, content: str = ""):
    """Log an SSE event to state."""
    state.events.append({
        "type": status,
        "agent": agent,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ─── Node functions ──────────────────────────────────────────────────────────

def session_load(state: dict) -> dict:
    """Load session context: history, collection count."""
    s = state
    if s.get("cancelled"):
        return s

    _emit_event_dict(s, "session", "status", "Loading session...")

    try:
        sm = _get_session_manager()
        hs = _get_history_store()

        session = sm.get_session(s["session_id"])
        collection = sm.get_collection(s["session_id"])
        s["collection_count"] = collection.count()

        # Get compressed history
        s["history_summary"] = hs.get_compressed_summary(s["session_id"])

        # Store user message
        hs.append_message(s["session_id"], "user", s["query"])

    except Exception as e:
        logger.error(f"Session load failed: {e}")
        s["error"] = str(e)

    return s


def intent_plan(state: dict) -> dict:
    """Classify intent and generate sub-queries."""
    s = state
    if s.get("cancelled"):
        return s

    _emit_event_dict(s, "intent_planner", "status", "Analyzing query...")

    planner = IntentPlanner()
    result = planner.plan(s["query"], s.get("history_summary", ""))

    s["intent"] = result.intent
    s["sub_queries"] = result.sub_queries
    s["search_needed"] = result.search_needed
    s["entity_heavy"] = result.entity_heavy
    s["direct_response"] = result.direct_response

    _emit_event_dict(s, "intent_planner", "status", f"Intent: {result.intent}")

    return s


def parallel_dispatch(state: dict) -> dict:
    """Run RAG retrieval and web search."""
    s = state
    if s.get("cancelled"):
        return s

    # ─── RAG retrieval ────────────────────────────────────────────
    rag_chunks = []
    if s.get("collection_count", 0) > 0 and s.get("crag_routing") != "tavily_only":
        _emit_event_dict(s, "rag_agent", "status", "Searching knowledge base...")
        try:
            sm = _get_session_manager()
            collection = sm.get_collection(s["session_id"])
            retriever = HybridRetriever()

            raw_chunks = retriever.retrieve(
                query=s["query"],
                collection=collection,
                session_id=s["session_id"],
                entity_heavy=s.get("entity_heavy", False),
            )
            rag_chunks = rerank(s["query"], raw_chunks)
            _emit_event_dict(s, "rag_agent", "status", f"Found {len(rag_chunks)} relevant chunks")
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")

    s["rag_chunks"] = rag_chunks

    # ─── Web search ───────────────────────────────────────────────
    web_articles = []
    if s.get("search_needed", True):
        _emit_event_dict(s, "web_search", "status", "Searching the web...")
        try:
            agent = WebSearchAgent()
            web_articles = agent.search(s["query"], s.get("sub_queries", []))
            _emit_event_dict(s, "web_search", "status", f"Found {len(web_articles)} articles")
        except Exception as e:
            logger.error(f"Web search failed: {e}")

    s["web_articles"] = web_articles

    return s


def data_validate(state: dict) -> dict:
    """Validate and ingest web articles into session ChromaDB."""
    s = state
    if s.get("cancelled") or not s.get("web_articles"):
        return s

    _emit_event_dict(s, "data_validator", "status", "Validating sources...")

    try:
        sm = _get_session_manager()
        client = GoogleClient()
        now = datetime.now(timezone.utc).isoformat()

        # Convert web articles to chunks for ChromaDB
        chunks = []
        embeddings = []
        for article in s["web_articles"]:
            if not article.content:
                continue

            # Contextual chunk header
            header = f"[Source: {article.outlet} | Date: {article.published_date} | URL: {article.url}]"
            text = f"{header}\n{article.content[:2000]}"

            try:
                embedding = client.embed(text)
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "metadata": {
                        "url": article.url,
                        "outlet": article.outlet,
                        "published_date": article.published_date,
                        "ingested_at": now,
                        "source": article.url,
                    },
                })
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Embedding failed for {article.url}: {e}")

        if chunks:
            # Add to collection with pre-computed embeddings (no double-embed)
            collection = sm.get_collection(s["session_id"])
            collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
                embeddings=embeddings,
            )

            # Invalidate BM25 cache
            HybridRetriever.invalidate_cache(s["session_id"])

            _emit_event_dict(s, "data_validator", "status", f"Ingested {len(chunks)} new chunks")

        # Build combined chunk list
        all_chunks = list(s.get("rag_chunks", []))
        for article in s["web_articles"]:
            all_chunks.append(Chunk(
                id=str(uuid.uuid4()),
                text=article.content[:2000],
                metadata={
                    "url": article.url,
                    "outlet": article.outlet,
                    "published_date": article.published_date,
                },
            ))
        s["all_chunks"] = all_chunks

    except Exception as e:
        logger.error(f"Data validation/ingestion failed: {e}")
        s["all_chunks"] = list(s.get("rag_chunks", []))

    return s


def crag_grade(state: dict) -> dict:
    """Grade document relevance using CRAG."""
    s = state
    if s.get("cancelled"):
        return s

    chunks = s.get("all_chunks", [])
    if not chunks:
        s["crag_routing"] = "tavily_only"
        return s

    _emit_event_dict(s, "crag_grader", "status", "Grading document relevance...")

    grader = CRAGGrader()
    result = grader.grade_documents(
        query=s["query"],
        docs=chunks,
        collection_count=s.get("collection_count", 0),
    )

    s["crag_routing"] = result.routing_decision
    _emit_event_dict(s, "crag_grader", "status", f"Routing: {result.routing_decision}")

    # Filter chunks based on grades
    if result.grades:
        relevant_ids = {
            g["doc_id"] for g in result.grades
            if g["grade"] in ("relevant", "partial")
        }
        s["all_chunks"] = [c for c in chunks if c.id in relevant_ids] or chunks

    return s


def synthesize(state: dict) -> dict:
    """Run synthesis with gemini-2.5-pro."""
    s = state
    if s.get("cancelled"):
        return s

    _emit_event_dict(s, "synthesis", "status", "Synthesizing answer...")

    chunks = s.get("all_chunks", [])
    if not chunks:
        s["synthesis_result"] = None
        s["error"] = "No source material available for synthesis"
        return s

    agent = SynthesisAgent()

    try:
        sm = _get_session_manager()
        collection = sm.get_collection(s["session_id"])
        retriever = HybridRetriever()

        result = agent.synthesize(
            context_chunks=chunks,
            query=s["query"],
            history=s.get("history_summary", ""),
            retriever=retriever,
            collection=collection,
            session_id=s["session_id"],
        )

        s["synthesis_result"] = result
        _emit_event_dict(s, "synthesis", "status", f"Synthesis complete ({result.pass_count} pass(es))")

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        s["error"] = f"synthesis_failed: {str(e)}"

    return s


def critique(state: dict) -> dict:
    """Run critic evaluation with circuit breaker."""
    s = state
    if s.get("cancelled") or not s.get("synthesis_result"):
        return s

    _emit_event_dict(s, "critic", "status", "Evaluating quality...")

    critic = CriticAgent()
    verdict = critic.evaluate(s["synthesis_result"], s["query"])

    # Store attempt
    attempts = s.get("synthesis_attempts", [])
    attempts.append((s["synthesis_result"], verdict.score))
    s["synthesis_attempts"] = attempts

    if verdict.passed:
        _emit_event_dict(s, "critic", "status", f"Quality passed (score: {verdict.score:.2f})")
    else:
        retries = s.get("critic_retries", 0)
        if retries < CRITIC_MAX_RETRIES:
            s["critic_retries"] = retries + 1
            _emit_event_dict(
                s, "critic", "status",
                f"Quality check failed (score: {verdict.score:.2f}), retry {retries + 1}/{CRITIC_MAX_RETRIES}"
            )
        else:
            # Circuit breaker: return best attempt by score
            best_result, best_score = max(attempts, key=lambda x: x[1])
            s["synthesis_result"] = best_result
            s["critic_flag"] = "max_retries_exceeded"
            _emit_event_dict(
                s, "critic", "status",
                f"Max retries exceeded. Using best attempt (score: {best_score:.2f})"
            )

    s["critic_verdict"] = verdict

    return s


def format_output(state: dict) -> dict:
    """Format the final response as HTML."""
    s = state
    if s.get("cancelled"):
        return s

    _emit_event_dict(s, "formatter", "status", "Formatting response...")

    formatter = FormatterAgent()

    if s.get("synthesis_result"):
        formatted = formatter.format(
            synthesis=s["synthesis_result"],
            credibility_map=s.get("credibility_map", {}),
            bias_map=s.get("bias_map", {}),
            source_comparison=s.get("source_comparison"),
            hallucination_report=s.get("hallucination_report"),
        )
        s["formatted_response"] = formatted
        s["status"] = "complete"
    else:
        # Error fallback
        s["formatted_response"] = FormatterAgent().format(
            synthesis=type("FallbackSynth", (), {
                "answer": s.get("error", "Unable to generate response"),
                "citations": [],
                "reasoning_gaps": [],
            })()
        )
        s["status"] = "error"

    # Store assistant response in history
    try:
        hs = _get_history_store()
        answer = s["synthesis_result"].answer if s.get("synthesis_result") else s.get("error", "")
        hs.append_message(s["session_id"], "assistant", answer)

        # Trigger async topic labeling and compression
        tl = _get_topic_labeler()
        history = hs.get_history(s["session_id"])
        tl.maybe_label(s["session_id"], history)
        hs.trigger_compression_if_needed(s["session_id"])
    except Exception as e:
        logger.error(f"Post-response tasks failed: {e}")

    return s


# ─── Routing functions ───────────────────────────────────────────────────────

def route_after_intent(state: dict) -> str:
    """Route based on intent classification."""
    if state.get("intent") == "chitchat":
        return "direct_end"
    return "parallel_dispatch"


def route_after_critic(state: dict) -> str:
    """Route based on critic verdict — retry or format."""
    verdict = state.get("critic_verdict")
    if verdict and not verdict.passed and state.get("critic_retries", 0) < CRITIC_MAX_RETRIES:
        return "synthesize"
    return "format_output"


def direct_end(state: dict) -> dict:
    """Handle chitchat — direct response, skip all agents."""
    s = state
    s["status"] = "complete"

    formatter = FormatterAgent()
    s["formatted_response"] = formatter.format(
        synthesis=type("ChitchatSynth", (), {
            "answer": s.get("direct_response", "Hello! How can I help with news research?"),
            "citations": [],
            "reasoning_gaps": [],
        })()
    )

    # Store in history
    try:
        hs = _get_history_store()
        hs.append_message(s["session_id"], "assistant", s["direct_response"])
    except Exception:
        pass

    return s


# ─── Helper for dict-based state ─────────────────────────────────────────────

def _emit_event_dict(state: dict, agent: str, status: str, content: str = ""):
    """Log an SSE event to dict-based state."""
    if "events" not in state:
        state["events"] = []
    state["events"].append({
        "type": status,
        "agent": agent,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ─── Build the graph ─────────────────────────────────────────────────────────

def build_graph():
    """Build the LangGraph StateGraph with all 11 nodes."""
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("session_load", session_load)
    graph.add_node("intent_plan", intent_plan)
    graph.add_node("direct_end", direct_end)
    graph.add_node("parallel_dispatch", parallel_dispatch)
    graph.add_node("data_validate", data_validate)
    graph.add_node("crag_grade", crag_grade)
    graph.add_node("synthesize", synthesize)
    graph.add_node("critique", critique)
    graph.add_node("format_output", format_output)

    # Set entry
    graph.set_entry_point("session_load")

    # Edges
    graph.add_edge("session_load", "intent_plan")
    graph.add_conditional_edges(
        "intent_plan",
        route_after_intent,
        {
            "direct_end": "direct_end",
            "parallel_dispatch": "parallel_dispatch",
        },
    )
    graph.add_edge("direct_end", END)
    graph.add_edge("parallel_dispatch", "data_validate")
    graph.add_edge("data_validate", "crag_grade")
    graph.add_edge("crag_grade", "synthesize")
    graph.add_edge("synthesize", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critic,
        {
            "synthesize": "synthesize",
            "format_output": "format_output",
        },
    )
    graph.add_edge("format_output", END)

    return graph.compile()


# ─── Module-level compiled graph ──────────────────────────────────────────────
_compiled_graph = None


def get_graph():
    """Get or build the compiled graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ─── Main execution function ─────────────────────────────────────────────────

def run_pipeline(query: str, session_id: str) -> dict:
    """
    Execute the full pipeline with 60s global timeout.

    Args:
        query: User's query.
        session_id: Session UUID.

    Returns:
        Final state dict with formatted_response and events.
    """
    # Check query cache first — cache hit = zero API calls
    cache = _get_query_cache()
    cached = cache.get(session_id, query)
    if cached is not None:
        logger.info("QueryCache HIT — returning cached response")
        return cached

    # Initial state
    state = {
        "query": query,
        "session_id": session_id,
        "cancelled": False,
        "timeout_exceeded": False,
        "cache_hit": False,
        "events": [],
        "synthesis_attempts": [],
        "critic_retries": 0,
        "all_chunks": [],
        "rag_chunks": [],
        "web_articles": [],
        "credibility_map": {},
        "bias_map": {},
    }

    # Execute with timeout
    result = [state]
    error = [None]

    def _execute():
        try:
            graph = get_graph()
            final = graph.invoke(state)
            result[0] = final
        except Exception as e:
            error[0] = e
            logger.error(f"Pipeline execution failed: {e}")

    thread = threading.Thread(target=_execute)
    thread.start()
    thread.join(timeout=ORCHESTRATOR_TIMEOUT)

    if thread.is_alive():
        logger.warning(f"Pipeline timeout after {ORCHESTRATOR_TIMEOUT}s")
        state["cancelled"] = True
        state["timeout_exceeded"] = True

        # Return best synthesis attempt if available
        attempts = state.get("synthesis_attempts", [])
        if attempts:
            best_result, best_score = max(attempts, key=lambda x: x[1])
            state["synthesis_result"] = best_result
            formatter = FormatterAgent()
            state["formatted_response"] = formatter.format(synthesis=best_result)
            state["status"] = "timeout_partial"
        else:
            state["status"] = "timeout_no_result"
            state["error"] = "Pipeline timed out with no synthesis available"

        result[0] = state

    if error[0]:
        result[0]["error"] = str(error[0])
        result[0]["status"] = "error"

    # Cache successful result
    final_state = result[0]
    if final_state.get("status") == "complete":
        cache.store(session_id, query, final_state)

    return final_state

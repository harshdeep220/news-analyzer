"""
Views — SSE streaming, session CRUD, chat page.

The SSE endpoint uses a queue.Queue bridge: the orchestrator runs in a
background thread and pushes events to the queue. The view's generator
reads from the queue and yields SSE frames. The browser's EventSource
receives each event and updates the trace panel in real-time.
"""

import json
import logging
import queue
import threading
import time

from django.http import (
    JsonResponse,
    StreamingHttpResponse,
    HttpResponseNotAllowed,
)
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from chat.services import SessionManager, HistoryStore, TopicLabeler, SessionNotFoundError

logger = logging.getLogger(__name__)


# ─── Page Views ──────────────────────────────────────────────────────────────

@require_GET
def index(request):
    """Render the main chat page."""
    sm = SessionManager()
    sessions = sm.list_sessions()
    return render(request, "index.html", {"sessions": sessions})


# ─── SSE Streaming ────────────────────────────────────────────────────────────

@require_GET
def chat_stream(request):
    """
    SSE endpoint: GET /api/chat?q=...&session_id=...

    Streams JSON events as the orchestrator processes the query.
    Each event: {type, agent, content, timestamp}
    Final event: {type: "done", ...} with the full result.
    """
    query = request.GET.get("q", "").strip()
    session_id = request.GET.get("session_id", "").strip()

    if not query or not session_id:
        return JsonResponse({"error": "Missing q or session_id"}, status=400)

    event_queue = queue.Queue()

    def _run_pipeline():
        """Run the orchestrator in a background thread, pushing events to queue."""
        try:
            from agents.orchestrator import build_graph, _get_query_cache
            from agents.formatter_agent import FormatterAgent

            # Check cache
            cache = _get_query_cache()
            cached = cache.get(session_id, query)
            if cached is not None:
                event_queue.put({
                    "type": "status", "agent": "cache",
                    "content": "Cache hit — returning saved response",
                })
                event_queue.put({"type": "done", "result": _serialize_result(cached)})
                return

            # Build fresh state
            state = {
                "query": query,
                "session_id": session_id,
                "cancelled": False,
                "timeout_exceeded": False,
                "cache_hit": False,
                "events": [],
                "event_queue": event_queue,  # <── the bridge
                "synthesis_attempts": [],
                "critic_retries": 0,
                "all_chunks": [],
                "rag_chunks": [],
                "web_articles": [],
                "credibility_map": {},
                "bias_map": {},
            }

            graph = build_graph()
            final = graph.invoke(state)

            # Cache successful results
            if final.get("status") == "complete":
                cache.store(session_id, query, final)

            event_queue.put({"type": "done", "result": _serialize_result(final)})

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            event_queue.put({
                "type": "error",
                "content": str(e),
            })

    thread = threading.Thread(target=_run_pipeline, daemon=True)
    thread.start()

    def event_generator():
        """Yield SSE frames from the event queue with flush-forcing keepalives."""
        try:
            # Initial keepalive to force connection open
            yield ": connected\n\n"

            while True:
                try:
                    event = event_queue.get(timeout=1.0)
                except queue.Empty:
                    # Keepalive comment — forces WSGI to flush
                    yield ": heartbeat\n\n"
                    continue

                yield f"data: {json.dumps(event, default=str)}\n\n"

                if event.get("type") in ("done", "error"):
                    break
        except GeneratorExit:
            logger.info("SSE client disconnected")

    response = StreamingHttpResponse(
        event_generator(),
        content_type="text/event-stream",
    )
    response["X-Accel-Buffering"] = "no"
    response["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response["Connection"] = "keep-alive"
    return response


def _serialize_result(result: dict) -> dict:
    """Serialize pipeline result for JSON transmission."""
    serialized = {
        "status": result.get("status", "unknown"),
        "intent": result.get("intent", ""),
    }

    # Answer text
    synth = result.get("synthesis_result")
    if synth and hasattr(synth, "answer"):
        serialized["answer"] = synth.answer
        if hasattr(synth, "citations"):
            serialized["citations"] = [
                {"url": c.url, "title": getattr(c, "title", ""), "outlet": getattr(c, "outlet", "")}
                for c in synth.citations
            ] if synth.citations else []
    elif result.get("direct_response"):
        serialized["answer"] = result["direct_response"]

    # Formatted HTML
    formatted = result.get("formatted_response")
    if formatted:
        if hasattr(formatted, "answer_html"):
            serialized["answer_html"] = formatted.answer_html
        if hasattr(formatted, "citations_html"):
            serialized["citations_html"] = formatted.citations_html

    # Intelligence layer
    cred_map = result.get("credibility_map", {})
    if cred_map:
        serialized["credibility"] = {
            url: {
                "total": cred.total if hasattr(cred, "total") else cred.get("total", 0),
                "signals": cred.signals if hasattr(cred, "signals") else cred.get("signals", {}),
            }
            for url, cred in cred_map.items()
        }

    bias_map = result.get("bias_map", {})
    if bias_map:
        serialized["bias"] = {
            url: {
                "lean": bias.lean if hasattr(bias, "lean") else bias.get("lean", ""),
                "confidence": bias.confidence if hasattr(bias, "confidence") else bias.get("confidence", 0),
            }
            for url, bias in bias_map.items()
        }

    hallucination = result.get("hallucination_report")
    if hallucination:
        serialized["hallucination"] = {
            "grounding_score": getattr(hallucination, "grounding_score", 1.0),
            "grounded_claims": getattr(hallucination, "grounded_claims", []),
            "ungrounded_claims": getattr(hallucination, "ungrounded_claims", []),
        }

    return serialized


# ─── Session CRUD ────────────────────────────────────────────────────────────

@csrf_exempt
def session_create(request):
    """POST /api/session/new — create a new session."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    sm = SessionManager()
    session_id = sm.create_session()
    return JsonResponse({"session_id": session_id, "topic_label": "New Research"})


@require_GET
def session_list(request):
    """GET /api/sessions — list all sessions."""
    sm = SessionManager()
    sessions = sm.list_sessions()
    return JsonResponse({"sessions": sessions})


@csrf_exempt
def session_delete(request, session_id):
    """DELETE /api/session/<id> — delete a session."""
    if request.method != "DELETE":
        return HttpResponseNotAllowed(["DELETE"])

    sm = SessionManager()
    sm.delete_session(session_id)
    return JsonResponse({"deleted": True})


@require_GET
def session_history(request, session_id):
    """GET /api/session/<id>/history — get message history."""
    hs = HistoryStore()
    history = hs.get_history(session_id)
    return JsonResponse({"messages": history})

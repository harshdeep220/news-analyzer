"""
CLI Testing Mode — Direct synchronous pipeline execution.
Bootstraps Django ORM before running.
"""

import sys
import os
import time

# Add project root AND web/ to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "web"))

# Bootstrap Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "newsrag.settings")
import django
django.setup()

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from chat.services import SessionManager, HistoryStore
from agents.orchestrator import build_graph, _get_query_cache


def highlight(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def run_sync(query: str, session_id: str) -> dict:
    """Run pipeline synchronously — no timeout thread, no recursion issues."""
    cache = _get_query_cache()
    cached = cache.get(session_id, query)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

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

    graph = build_graph()
    final = graph.invoke(state)

    if final.get("status") == "complete":
        cache.store(session_id, query, final)

    return final


def print_result(result: dict):
    """Pretty-print the pipeline result."""
    status = result.get("status", "unknown")

    agents_run = []
    for e in result.get("events", []):
        agent = e.get("agent", "")
        if agent not in agents_run:
            agents_run.append(agent)
    if agents_run:
        print(highlight(f"  Agents: {' → '.join(agents_run)}", "35"))

    if result.get("cache_hit"):
        print(highlight("  ⚡ Cache hit — zero API calls", "33"))

    if status in ("complete", "timeout_partial"):
        synth = result.get("synthesis_result")
        if synth and hasattr(synth, "answer"):
            print(f"\n{highlight('A:', '34;1')} {synth.answer}")
        elif result.get("direct_response"):
            print(f"\n{highlight('A:', '34;1')} {result['direct_response']}")
    elif status == "error":
        print(highlight(f"\n  [Error] {result.get('error', 'Unknown')}", "31"))
    else:
        print(highlight(f"\n  [Status: {status}]", "33"))

    cred_map = result.get("credibility_map", {})
    if cred_map:
        print(highlight("\n  [Credibility]", "33"))
        for url, cred in list(cred_map.items())[:3]:
            total = cred.total if hasattr(cred, "total") else cred.get("total", 0)
            print(f"    {url[:55]}  →  {total}/100")

    bias_map = result.get("bias_map", {})
    if bias_map:
        print(highlight("\n  [Bias]", "33"))
        for url, bias in list(bias_map.items())[:3]:
            lean = bias.lean if hasattr(bias, "lean") else bias.get("lean", "?")
            if lean != "insufficient_content":
                conf = bias.confidence if hasattr(bias, "confidence") else bias.get("confidence", 0)
                print(f"    {url[:55]}  →  {lean} ({conf:.0%})")

    hallucination = result.get("hallucination_report")
    if hallucination:
        ungrounded = getattr(hallucination, "ungrounded_claims", [])
        score = getattr(hallucination, "grounding_score", 1.0)
        if ungrounded:
            print(highlight(f"\n  [Grounding: {score:.0%}] {len(ungrounded)} unverified claim(s):", "31"))
            for c in ungrounded[:3]:
                print(highlight(f"    ⚠ {c}", "31"))


def main():
    os.system("color")

    print(highlight("\n ═══ News RAG — CLI ═══", "36;1"))
    print(highlight(" Ctrl+C to cancel. Type 'exit' to quit.\n", "90"))

    sm = SessionManager()
    sid = sm.create_session()
    print(highlight(f" Session: {sid}\n", "90"))

    try:
        while True:
            try:
                query = input(highlight("You: ", "32;1"))
            except EOFError:
                break

            if query.strip().lower() in ("exit", "quit", "q"):
                break
            if not query.strip():
                continue

            try:
                start = time.time()
                result = run_sync(query, sid)
                elapsed = time.time() - start

                print(highlight(f"\n  [{elapsed:.1f}s]", "90"), end="")
                print_result(result)
                print()

            except KeyboardInterrupt:
                print(highlight("\n  Cancelled.\n", "33"))

    except KeyboardInterrupt:
        pass
    finally:
        print(highlight("\n Goodbye!\n", "36"))
        sm.delete_session(sid)


if __name__ == "__main__":
    main()

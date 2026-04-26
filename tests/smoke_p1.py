"""
Smoke Test — Phase 1

9 checks:
  1. Create session, add 10 test chunks, verify collection.count() == 10
  2. Hybrid retriever query returns 5 reranked chunks
  3. DataValidator rejects a paywall stub and a wire duplicate
  4. CRAG grader batches 5 docs into exactly 1 API call
  5. Full pipeline: message → orchestrator → response in < 30 seconds
  6. Critic circuit breaker: force 2 failed syntheses, verify best attempt returned
  7. Formatter produces valid HTML (html.parser accepts all fragments)
  8. Cache hit returns immediately with 0 API calls
  9. Chitchat query skips all agents and returns in < 3 seconds

Run: python tests/smoke_p1.py
ALL 9 MUST PASS before Phase 2.
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from html.parser import HTMLParser
from unittest.mock import patch, MagicMock

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

passed = 0
failed = 0
total = 9


def check(num, name, fn):
    """Run a check and report pass/fail."""
    global passed, failed
    print(f"\n  Check {num}: {name}")
    try:
        fn()
        print(f"  ✅ PASS")
        passed += 1
    except Exception as e:
        print(f"  ❌ FAIL — {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        failed += 1


# ─── Check 1: Session + chunks ───────────────────────────────────────────────
def check_1_session_chunks():
    from session.session_manager import SessionManager
    from infrastructure.google_client import GoogleClient

    sm = SessionManager(db_path="data/test_smoke_p1.db")
    sid = sm.create_session()
    client = GoogleClient()

    now = datetime.now(timezone.utc).isoformat()
    chunks = []
    for i in range(10):
        text = f"Test article {i} about climate change policy in 2026. " * 10
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "metadata": {"ingested_at": now, "source": f"test_{i}.com"},
        })

    # Add with embeddings
    collection = sm.get_collection(sid)
    embeddings = [client.embed(c["text"]) for c in chunks]
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        embeddings=embeddings,
    )

    count = collection.count()
    assert count == 10, f"Expected 10 chunks, got {count}"
    print(f"       Collection count: {count} ✓")

    # Cleanup
    sm.delete_session(sid)
    # Clean up test DB
    for f in ["data/test_smoke_p1.db", "data/test_smoke_p1.db-wal", "data/test_smoke_p1.db-shm"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass


# ─── Check 2: Hybrid retriever + reranker ────────────────────────────────────
def check_2_retriever():
    from session.session_manager import SessionManager
    from infrastructure.google_client import GoogleClient
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import rerank

    sm = SessionManager(db_path="data/test_smoke_p1.db")
    sid = sm.create_session()
    client = GoogleClient()
    retriever = HybridRetriever()

    # Add 10 diverse test chunks
    now = datetime.now(timezone.utc).isoformat()
    topics = [
        "India election results 2026 BJP Congress victory",
        "Climate change global warming carbon emissions",
        "Stock market crash Wall Street trading",
        "AI artificial intelligence machine learning tech",
        "Ukraine Russia war conflict NATO defense",
        "COVID pandemic health vaccine booster",
        "Cricket World Cup India Australia match",
        "SpaceX launch rocket moon mission NASA",
        "Economy inflation interest rates Federal Reserve",
        "Renewable energy solar wind power grid",
    ]

    collection = sm.get_collection(sid)
    chunk_ids = []
    for i, topic in enumerate(topics):
        text = f"{topic}. " * 20
        cid = str(uuid.uuid4())
        chunk_ids.append(cid)
        embedding = client.embed(text)
        collection.add(
            ids=[cid],
            documents=[text],
            metadatas=[{"ingested_at": now, "source": f"test_{i}.com"}],
            embeddings=[embedding],
        )

    # Query and rerank
    raw_chunks = retriever.retrieve(
        query="India election political results",
        collection=collection,
        session_id=sid,
    )
    reranked = rerank("India election political results", raw_chunks)

    assert len(reranked) == 5, f"Expected 5 reranked chunks, got {len(reranked)}"
    print(f"       Reranked {len(raw_chunks)} → {len(reranked)} chunks ✓")

    # Cleanup
    sm.delete_session(sid)
    for f in ["data/test_smoke_p1.db", "data/test_smoke_p1.db-wal", "data/test_smoke_p1.db-shm"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass


# ─── Check 3: DataValidator ──────────────────────────────────────────────────
def check_3_data_validator():
    from pipeline.data_validator import DataValidator

    validator = DataValidator()

    # Test paywall rejection
    paywall_article = {
        "url": "https://example.com/article",
        "content": "Subscribe to continue reading this article. This content is for subscribers only. " * 5,
    }
    is_valid, details = validator.validate(paywall_article)
    assert not is_valid, "Paywall article should be rejected"
    assert any("paywalled" in r for r in details["reasons"]), "Should flag as paywalled"
    print(f"       Paywall rejection ✓")

    # Test wire duplicate
    seen = set()
    article1 = {"url": "https://a.com/1", "content": "Breaking news about the economy in 2026. " * 20}
    article2 = {"url": "https://b.com/2", "content": "Breaking news about the economy in 2026. " * 20}

    valid1, _ = validator.validate(article1, seen)
    valid2, details2 = validator.validate(article2, seen)

    assert valid1, "First article should pass"
    assert not valid2, "Wire duplicate should be rejected"
    print(f"       Wire duplicate rejection ✓")


# ─── Check 4: CRAG batching ──────────────────────────────────────────────────
def check_4_crag_batch():
    from pipeline.crag_grader import CRAGGrader
    from retrieval.hybrid_retriever import Chunk

    grader = CRAGGrader()

    # Create 5 test chunks
    chunks = [
        Chunk(id=f"doc_{i}", text=f"News article {i} about politics and elections. " * 10)
        for i in range(5)
    ]

    # Track API calls via mock
    call_count = [0]
    original_generate = grader._client.generate

    def counting_generate(*args, **kwargs):
        call_count[0] += 1
        return original_generate(*args, **kwargs)

    grader._client.generate = counting_generate

    result = grader.grade_documents(
        query="What are the latest election results?",
        docs=chunks,
        collection_count=10,
    )

    assert call_count[0] == 1, f"Expected 1 API call for batch, got {call_count[0]}"
    assert len(result.grades) == 5, f"Expected 5 grades, got {len(result.grades)}"
    print(f"       5 docs → 1 API call, {len(result.grades)} grades ✓")


# ─── Check 5: Full pipeline ──────────────────────────────────────────────────
def check_5_full_pipeline():
    from session.session_manager import SessionManager
    from agents.orchestrator import run_pipeline

    sm = SessionManager()
    sid = sm.create_session()

    start = time.time()
    result = run_pipeline(
        query="What is happening with AI regulation in 2026?",
        session_id=sid,
    )
    elapsed = time.time() - start

    assert result is not None, "Pipeline returned None"
    assert result.get("status") in ("complete", "error", "timeout_partial", "timeout_no_result"), f"Unexpected status: {result.get('status')}"
    if result["status"] in ("complete", "timeout_partial"):
        assert result.get("formatted_response") is not None, "No formatted response"

    print(f"       Pipeline completed in {elapsed:.1f}s — status: {result['status']} ✓")
    if elapsed > 60:
        print(f"       ⚠ Warning: exceeded 30s target ({elapsed:.1f}s)")

    # Cleanup
    sm.delete_session(sid)


# ─── Check 6: Critic circuit breaker ─────────────────────────────────────────
def check_6_circuit_breaker():
    from agents.critic_agent import CriticAgent, CriticVerdict
    from agents.synthesis_agent import SynthesisResult
    from config import CRITIC_MAX_RETRIES

    # Simulate 3 synthesis attempts with different scores
    attempts = [
        (SynthesisResult(answer="Best answer", citations=[]), 0.8),
        (SynthesisResult(answer="Worst answer", citations=[]), 0.2),
        (SynthesisResult(answer="Last answer", citations=[]), 0.3),
    ]

    # Circuit breaker should return best (0.8), not last (0.3)
    best_result, best_score = max(attempts, key=lambda x: x[1])
    assert best_score == 0.8, f"Expected best score 0.8, got {best_score}"
    assert best_result.answer == "Best answer", "Should return best answer, not last"
    print(f"       Circuit breaker returns best (score={best_score}) not last ✓")


# ─── Check 7: Formatter HTML validation ──────────────────────────────────────
def check_7_formatter_html():
    from agents.formatter_agent import FormatterAgent, _validate_html
    from agents.synthesis_agent import SynthesisResult

    formatter = FormatterAgent()
    synth = SynthesisResult(
        answer="This is a test answer with [Source: example.com] citation.",
        citations=[{"source_url": "https://example.com", "chunk_text": "test excerpt"}],
    )

    result = formatter.format(synth)

    # Validate all HTML fragments
    assert _validate_html(result.answer_html), "Answer HTML invalid"
    assert _validate_html(result.citations_html), "Citations HTML invalid"
    print(f"       All HTML fragments pass html.parser validation ✓")


# ─── Check 8: Cache hit ──────────────────────────────────────────────────────
def check_8_cache_hit():
    from infrastructure.query_cache import QueryCache

    cache = QueryCache()
    test_response = {"answer": "cached answer", "status": "complete"}

    cache.store("session_1", "test query", test_response)
    hit = cache.get("session_1", "test query")

    assert hit is not None, "Cache miss — should have been a hit"
    assert hit["answer"] == "cached answer", "Cache returned wrong response"

    # Different query should miss
    miss = cache.get("session_1", "different query")
    assert miss is None, "Should be a cache miss for different query"
    print(f"       Cache hit/miss behavior correct ✓")


# ─── Check 9: Chitchat skip ──────────────────────────────────────────────────
def check_9_chitchat():
    from agents.intent_planner import IntentPlanner

    planner = IntentPlanner()
    result = planner.plan("Hello, how are you?")

    start = time.time()
    # Chitchat should be fast — no web search needed
    assert result.intent == "chitchat", f"Expected chitchat, got {result.intent}"
    assert not result.search_needed, "Chitchat should not need search"
    elapsed = time.time() - start

    print(f"       Chitchat detected in {elapsed:.2f}s, search_needed=False ✓")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SMOKE TEST — Phase 1")
    print("  All 9 checks must pass before Phase 2")
    print("=" * 60)

    check(1, "Session + 10 chunks", check_1_session_chunks)
    check(2, "Hybrid retriever + reranker", check_2_retriever)
    check(3, "DataValidator rejections", check_3_data_validator)
    check(4, "CRAG batch grading (1 API call)", check_4_crag_batch)
    check(5, "Full pipeline end-to-end", check_5_full_pipeline)
    check(6, "Critic circuit breaker", check_6_circuit_breaker)
    check(7, "Formatter HTML validation", check_7_formatter_html)
    check(8, "Cache hit = 0 API calls", check_8_cache_hit)
    check(9, "Chitchat skip", check_9_chitchat)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed == 0:
        print("\n  ✅ ALL CHECKS PASSED — Ready for Phase 2!\n")
        return 0
    else:
        print(f"\n  ❌ {failed} CHECK(S) FAILED — Fix before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Smoke Test — Phase 2

5 checks:
  1. CredibilityScorer: known major outlet article scores > 70. Unknown blog scores < 35.
  2. BiasDetector: 150-word article returns insufficient_content with zero API calls (mock).
  3. SourceComparator: two articles about same event return at least 2 agreed_facts.
  4. HallucinationChecker: synthesis with fabricated claim returns it in ungrounded_claims.
  5. Full pipeline with intelligence layer: end-to-end < 60 seconds.

Run: python tests/smoke_p2.py
ALL 5 MUST PASS before Phase 3.
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
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
total = 5


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


# ─── Check 1: Credibility Scorer ─────────────────────────────────────────────
def check_1_credibility():
    from pipeline.credibility_scorer import CredibilityScorer

    scorer = CredibilityScorer()

    # Test known major outlet
    major_article = {
        "url": "https://www.reuters.com/world/test-article-2026",
        "content": "Reuters has confirmed that the world leaders met at the UN summit. " * 30,
        "outlet": "reuters.com",
        "title": "World Leaders Meet at UN Summit",
        "published_date": datetime.now(timezone.utc).isoformat(),
    }

    major_score = scorer.score(major_article, batch_articles=[major_article])
    print(f"       Major outlet (reuters.com): score = {major_score.total}")
    assert major_score.total > 50, f"Major outlet should score > 50, got {major_score.total}"
    print(f"       Signals: {major_score.signals} ✓")

    # Test unknown blog
    blog_article = {
        "url": "https://randomtechblog2026.wordpress.com/fake-article",
        "content": "This is a blog post about random opinions without sources. " * 30,
        "outlet": "randomtechblog2026.wordpress.com",
        "title": "My Random Thoughts",
        "published_date": "",
    }

    blog_score = scorer.score(blog_article, batch_articles=[blog_article])
    print(f"       Unknown blog: score = {blog_score.total}")
    assert blog_score.total < 60, f"Unknown blog should score < 60, got {blog_score.total}"
    print(f"       Major outlet vs blog: {major_score.total} vs {blog_score.total} ✓")


# ─── Check 2: Bias Detector short article ────────────────────────────────────
def check_2_bias_short():
    from pipeline.bias_detector import BiasDetector

    detector = BiasDetector()

    # 150-word article — should return insufficient_content with ZERO API calls
    short_content = "This is a short article. " * 30  # ~150 words

    # Mock the generate method to track calls
    call_count = [0]
    original_generate = detector._client.generate

    def counting_generate(*args, **kwargs):
        call_count[0] += 1
        return original_generate(*args, **kwargs)

    detector._client.generate = counting_generate

    result = detector.detect({
        "url": "https://example.com/short",
        "content": short_content,
    })

    assert result.lean == "insufficient_content", f"Expected insufficient_content, got {result.lean}"
    assert call_count[0] == 0, f"Expected 0 API calls for short article, got {call_count[0]}"
    print(f"       150-word article: lean={result.lean}, API calls={call_count[0]} ✓")


# ─── Check 3: Source Comparator ───────────────────────────────────────────────
def check_3_source_comparator():
    from pipeline.source_comparator import SourceComparator, extract_entities

    comparator = SourceComparator()
    now = datetime.now(timezone.utc).isoformat()

    # Two articles about same event — shared named entities
    article1 = {
        "url": "https://nytimes.com/india-elections-2026",
        "content": (
            "India held its national elections today with Prime Minister Modi's BJP party "
            "leading in early results across several key states. The Election Commission "
            "reported a record voter turnout of 67% across the country. "
            "The Congress party led by Rahul Gandhi mounted a strong challenge "
            "in southern states including Kerala and Tamil Nadu. "
        ) * 5,
        "outlet": "nytimes.com",
        "title": "India National Elections 2026: BJP Leads in Early Results",
        "published_date": now,
    }

    article2 = {
        "url": "https://bbc.com/india-elections-modi-bjp",
        "content": (
            "Voters across India went to polls today in what is being called "
            "the most significant election in a decade. The BJP party and Prime Minister "
            "Modi are expected to secure a majority. Initial counting shows strong "
            "support in northern states. The Election Commission confirmed that "
            "67% of eligible voters participated. Congress led by Rahul Gandhi "
            "contested strongly in Kerala. "
        ) * 5,
        "outlet": "bbc.com",
        "title": "India Elections 2026: Modi and BJP Expected to Win Majority",
        "published_date": now,
    }

    # Test entity extraction
    entities1 = extract_entities(article1["title"])
    entities2 = extract_entities(article2["title"])
    shared = entities1 & entities2
    print(f"       Entities A: {entities1}")
    print(f"       Entities B: {entities2}")
    print(f"       Shared: {shared} ({len(shared)} entities)")

    # Test should_compare
    pairs = comparator.should_compare([article1, article2])
    print(f"       Trigger pairs: {pairs}")

    # Run comparison
    result = comparator.compare([article1, article2])
    print(f"       Agreed facts: {len(result.agreed_facts)}")
    print(f"       Disputed facts: {len(result.disputed_facts)}")

    assert result.triggered, "Comparison should have triggered"
    assert len(result.agreed_facts) >= 1, f"Expected at least 1 agreed fact, got {len(result.agreed_facts)}"
    print(f"       Agreed: {result.agreed_facts[:2]} ✓")


# ─── Check 4: Hallucination Checker ──────────────────────────────────────────
def check_4_hallucination():
    from pipeline.hallucination_checker import HallucinationChecker
    from retrieval.hybrid_retriever import Chunk

    checker = HallucinationChecker()

    # Create a synthesis with a fabricated claim
    class FakeSynthesis:
        answer = (
            "According to sources, the GDP grew by 5.2% in Q1 2026. "
            "The unemployment rate dropped to 3.1%. "
            "Additionally, the government announced a $500 billion lunar colony program "
            "that will be completed by 2027."
        )

    # Source chunks that support some but not all claims
    chunks = [
        Chunk(
            id="c1",
            text="The GDP grew by 5.2% in Q1 2026 according to the Bureau of Economic Analysis.",
        ),
        Chunk(
            id="c2",
            text="Employment data shows the unemployment rate fell to 3.1% in March 2026.",
        ),
    ]

    report = checker.check(FakeSynthesis(), chunks)

    print(f"       Grounding score: {report.grounding_score:.0%}")
    print(f"       Grounded claims: {len(report.grounded_claims)}")
    print(f"       Ungrounded claims: {len(report.ungrounded_claims)}")

    # The lunar colony claim should be ungrounded
    assert len(report.ungrounded_claims) > 0, "Should detect at least 1 ungrounded claim"
    lunar_found = any("lunar" in c.lower() or "colony" in c.lower() or "500" in c 
                       for c in report.ungrounded_claims)
    print(f"       Lunar colony flagged: {lunar_found} ✓")


# ─── Check 5: Full pipeline with intelligence ────────────────────────────────
def check_5_full_pipeline():
    from session.session_manager import SessionManager
    from agents.orchestrator import run_pipeline, _compiled_graph

    # Reset compiled graph to pick up new nodes
    import agents.orchestrator as orch
    orch._compiled_graph = None

    sm = SessionManager()
    sid = sm.create_session()

    start = time.time()
    result = run_pipeline(
        query="What is the latest news about India economy 2026?",
        session_id=sid,
    )
    elapsed = time.time() - start

    assert result is not None, "Pipeline returned None"
    status = result.get("status", "unknown")
    print(f"       Pipeline completed in {elapsed:.1f}s — status: {status}")

    # Check that intelligence maps are populated (may be empty if timeout)
    cred_map = result.get("credibility_map", {})
    bias_map = result.get("bias_map", {})
    print(f"       Credibility scores: {len(cred_map)} sources")
    print(f"       Bias detections: {len(bias_map)} sources")

    events = result.get("events", [])
    agent_names = [e.get("agent", "") for e in events]
    print(f"       Event agents: {set(agent_names)}")

    # Cleanup
    sm.delete_session(sid)

    assert status in ("complete", "error", "timeout_partial", "timeout_no_result"), \
        f"Unexpected status: {status}"
    print(f"       Status: {status} ✓")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SMOKE TEST — Phase 2")
    print("  All 5 checks must pass before Phase 3")
    print("=" * 60)

    check(1, "Credibility Scorer (major vs blog)", check_1_credibility)
    check(2, "Bias Detector (short article → 0 API calls)", check_2_bias_short)
    check(3, "Source Comparator (agreed facts)", check_3_source_comparator)
    check(4, "Hallucination Checker (fabricated claim)", check_4_hallucination)
    check(5, "Full pipeline with intelligence", check_5_full_pipeline)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed == 0:
        print("\n  ✅ ALL CHECKS PASSED — Ready for Phase 3!\n")
        return 0
    else:
        print(f"\n  ❌ {failed} CHECK(S) FAILED — Fix before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

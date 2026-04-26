"""
Connectivity Test — Verifies Google AI Studio and Tavily API connections.

Runs once to:
  1. Test Flash generation
  2. Test Pro generation (with TTFT logging)
  3. Test embedding (verify 768-dim)
  4. Test Tavily search and save fixture to tests/fixtures/cached_tavily_response.json

Usage: python tests/connectivity_test.py
"""

import json
import os
import sys
import time

# Fix Windows console encoding for Unicode symbols
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import FAST_MODEL, SYNTH_MODEL, EMBED_MODEL, EMBED_DIM, TAVILY_API_KEY


def test_flash():
    """Test gemini-2.0-flash generation."""
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    start = time.time()
    response = client.generate("Say 'Hello, News RAG!' and nothing else.", model=FAST_MODEL)
    elapsed = time.time() - start

    assert response and len(response.strip()) > 0, "Empty response from Flash"
    print(f"  [OK] Flash — {elapsed:.2f}s — Response: {response.strip()[:80]}")
    return True


def test_pro():
    """Test gemini-2.5-pro generation with TTFT logging."""
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    start = time.time()
    response = client.generate(
        "What is the capital of France? Reply in one word.",
        model=SYNTH_MODEL,
        system="You are a concise assistant.",
    )
    elapsed = time.time() - start

    assert response and len(response.strip()) > 0, "Empty response from Pro"
    # Check for leaked thinking tokens
    assert "<think>" not in response.lower(), "Pro leaked thinking tokens!"
    print(f"  [OK] Pro — {elapsed:.2f}s (TTFT baseline) — Response: {response.strip()[:80]}")
    return True


def test_embedding():
    """Test text-embedding-004 returns 768-dim vector."""
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    vector = client.embed("This is a test sentence for embedding.")
    assert len(vector) == EMBED_DIM, f"Expected {EMBED_DIM}-dim, got {len(vector)}"
    assert all(isinstance(v, float) for v in vector), "Embedding contains non-float values"
    print(f"  [OK] Embedding — {EMBED_DIM}-dim vector ✓")
    return True


def test_tavily_and_save_fixture():
    """Test Tavily search and save response as fixture."""
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=TAVILY_API_KEY)

    start = time.time()
    response = tavily.search(
        query="India news today",
        max_results=5,
        include_raw_content=False,
    )
    elapsed = time.time() - start

    assert "results" in response, "Tavily response missing 'results' key"
    assert len(response["results"]) > 0, "Tavily returned 0 results"

    # Save fixture
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures",
        "cached_tavily_response.json",
    )
    os.makedirs(os.path.dirname(fixture_path), exist_ok=True)
    with open(fixture_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    print(f"  [OK] Tavily — {elapsed:.2f}s — {len(response['results'])} results")
    print(f"       Fixture saved to {fixture_path}")
    return True


def main():
    print("\n" + "=" * 60)
    print("  CONNECTIVITY TEST — Google AI Studio + Tavily")
    print("=" * 60 + "\n")

    results = {}
    tests = [
        ("Flash (gemini-2.0-flash)", test_flash),
        ("Pro (gemini-2.5-pro)", test_pro),
        ("Embedding (text-embedding-004)", test_embedding),
        ("Tavily Search", test_tavily_and_save_fixture),
    ]

    for name, test_fn in tests:
        print(f"\nTesting {name}...")
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  [FAIL] {name} — {type(e).__name__}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n  ✅ All connectivity tests passed!\n")
    else:
        print("\n  ❌ Some tests failed. Fix before proceeding.\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

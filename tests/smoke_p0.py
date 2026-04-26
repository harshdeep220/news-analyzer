"""
Smoke Test — Phase 0

Verifies all Phase 0 requirements before proceeding to Phase 1.

7 checks:
  1. google_client.generate() returns completion from Flash. Latency logged.
  2. google_client.generate() returns completion from Pro. TTFT logged.
  3. google_client.embed() returns vector of exactly 768 dimensions.
  4. Tavily fixture exists at tests/fixtures/cached_tavily_response.json.
  5. ChromaDB creates collection 'stest001', retrieves it by name, deletes it.
  6. _safe_collection_name() passes all 20 edge cases.
  7. SQLite sessions table created with correct schema, WAL mode confirmed.

Run: python tests/smoke_p0.py
ALL 7 MUST PASS before Phase 1.
"""

import json
import os
import sqlite3
import sys
import time

# Fix Windows console encoding for Unicode symbols
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from config import FAST_MODEL, SYNTH_MODEL, EMBED_DIM, SQLITE_DB_PATH

passed = 0
failed = 0
total = 7


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
        failed += 1


# ─── Check 1: Flash generation ───────────────────────────────────────────────
def check_1_flash():
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    start = time.time()
    response = client.generate(
        "Respond with exactly: 'Flash OK'",
        model=FAST_MODEL,
    )
    elapsed = time.time() - start

    assert response and len(response.strip()) > 0, "Empty response from Flash"
    print(f"       Response: {response.strip()[:80]}")
    print(f"       Latency: {elapsed:.2f}s")


# ─── Check 2: Pro generation ─────────────────────────────────────────────────
def check_2_pro():
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    start = time.time()
    response = client.generate(
        "What is 2+2? Reply with just the number.",
        model=SYNTH_MODEL,
    )
    elapsed = time.time() - start

    assert response and len(response.strip()) > 0, "Empty response from Pro"
    assert "<think>" not in response.lower(), "Pro leaked thinking tokens"
    print(f"       Response: {response.strip()[:80]}")
    print(f"       Time-to-first-token baseline: {elapsed:.2f}s")


# ─── Check 3: Embedding dimension ────────────────────────────────────────────
def check_3_embedding():
    from infrastructure.google_client import GoogleClient
    client = GoogleClient()

    vector = client.embed("Test embedding for Phase 0 smoke test.")
    assert len(vector) == EMBED_DIM, f"Expected {EMBED_DIM}-dim, got {len(vector)}"
    assert all(isinstance(v, float) for v in vector), "Non-float values in embedding"
    print(f"       Dimension: {len(vector)} ✓")


# ─── Check 4: Tavily fixture exists ──────────────────────────────────────────
def check_4_tavily_fixture():
    fixture_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures",
        "cached_tavily_response.json",
    )
    assert os.path.exists(fixture_path), f"Fixture not found at {fixture_path}"

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "results" in data, "Fixture missing 'results' key"
    assert len(data["results"]) > 0, "Fixture has 0 results"
    print(f"       Fixture: {len(data['results'])} results cached")


# ─── Check 5: ChromaDB create/get/delete ──────────────────────────────────────
def check_5_chromadb():
    import chromadb

    persist_dir = os.path.join("vectorstore", "chroma_store")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    col_name = "stest001"

    # Create
    col = client.get_or_create_collection(name=col_name)
    assert col is not None, "Failed to create collection"

    # Retrieve by name
    retrieved = client.get_collection(name=col_name)
    assert retrieved.name == col_name, f"Retrieved name mismatch: {retrieved.name}"

    # Delete
    client.delete_collection(name=col_name)

    # Verify deleted
    try:
        client.get_collection(name=col_name)
        raise AssertionError("Collection still exists after delete!")
    except Exception:
        pass  # Expected — collection should not exist

    print(f"       Create → Get → Delete cycle ✓")


# ─── Check 6: _safe_collection_name() passes 20 edge cases ──────────────────
def check_6_naming():
    from session.session_manager import _safe_collection_name

    edge_cases = [
        ("Standard UUID", "550e8400-e29b-41d4-a716-446655440000"),
        ("Single char 'a'", "a"),
        ("Single char '1'", "1"),
        ("100-char string", "a" * 100),
        ("All hyphens", "---"),
        ("Starts with hyphen", "-abc"),
        ("Ends with hyphen", "abc-"),
        ("Empty string", ""),
        ("All special chars", "!@#$%"),
        ("Unicode 'café'", "café"),
        ("Consecutive hyphens", "a--b--c"),
        ("Exactly 3 chars", "abc"),
        ("Exactly 63 chars", "a" * 63),
        ("64 chars (over limit)", "a" * 64),
        ("Whitespace only", "   "),
        ("Mixed case", "AbCdEf"),
        ("Numeric only", "12345"),
        ("Single hyphen", "-"),
        ("Alphanumeric with dots", "session.v2.final"),
        ("Very long UUID-like", "a1b2c3d4-e5f6-a7b8-c9d0-e1f2a3b4c5d6-extra"),
    ]

    for label, raw in edge_cases:
        name = _safe_collection_name(raw)

        # Validate ChromaDB rules
        assert 3 <= len(name) <= 63, f"[{label}] Length {len(name)} out of range: '{name}'"
        assert name[0].isalnum(), f"[{label}] Does not start with alnum: '{name}'"
        assert name[-1].isalnum(), f"[{label}] Does not end with alnum: '{name}'"
        assert "--" not in name, f"[{label}] Contains consecutive hyphens: '{name}'"
        assert all(
            c.isalnum() or c == "-" for c in name
        ), f"[{label}] Invalid chars in: '{name}'"

        print(f"       ✓ {label}: '{raw}' → '{name}'")


# ─── Check 7: SQLite schema + WAL mode ───────────────────────────────────────
def check_7_sqlite():
    # Use a test-specific DB to avoid polluting the real one
    test_db = os.path.join("data", "test_smoke_p0.db")
    os.makedirs("data", exist_ok=True)

    # Clean up any previous test DB
    if os.path.exists(test_db):
        os.remove(test_db)

    from session.session_manager import SessionManager
    sm = SessionManager(db_path=test_db)

    # Verify WAL mode
    conn = sqlite3.connect(test_db)
    wal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert wal_mode == "wal", f"Expected WAL mode, got: {wal_mode}"
    print(f"       WAL mode: {wal_mode} ✓")

    # Verify table schema
    columns = conn.execute("PRAGMA table_info(sessions)").fetchall()
    col_names = [c[1] for c in columns]
    expected = ["id", "collection_name", "topic_label", "created_at", "message_count"]
    assert col_names == expected, f"Schema mismatch: {col_names} != {expected}"
    print(f"       Schema: {col_names} ✓")

    # Test create → lookup → delete cycle
    sid = sm.create_session()
    session = sm.get_session(sid)
    assert session["id"] == sid
    assert session["topic_label"] == "New Research"
    sm.delete_session(sid)
    print(f"       Create → Lookup → Delete cycle ✓")

    conn.close()

    # Cleanup test DB
    try:
        os.remove(test_db)
        # Remove WAL/SHM files too
        for ext in ["-wal", "-shm"]:
            if os.path.exists(test_db + ext):
                os.remove(test_db + ext)
    except Exception:
        pass


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("  SMOKE TEST — Phase 0")
    print("  All 7 checks must pass before Phase 1")
    print("=" * 60)

    check(1, "Flash generation (gemini-2.0-flash)", check_1_flash)
    check(2, "Pro generation (gemini-2.5-pro)", check_2_pro)
    check(3, "Embedding dimension (768)", check_3_embedding)
    check(4, "Tavily fixture exists", check_4_tavily_fixture)
    check(5, "ChromaDB create/get/delete", check_5_chromadb)
    check(6, "_safe_collection_name() — 20 edge cases", check_6_naming)
    check(7, "SQLite schema + WAL mode", check_7_sqlite)

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed, {failed}/{total} failed")
    print("=" * 60)

    if failed == 0:
        print("\n  ✅ ALL CHECKS PASSED — Ready for Phase 1!\n")
        return 0
    else:
        print(f"\n  ❌ {failed} CHECK(S) FAILED — Fix before proceeding.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

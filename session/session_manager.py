"""
Session Manager — Creates, loads, and deletes research sessions.

Responsibilities:
  - ChromaDB collection naming via _safe_collection_name()
  - SQLite session_id → collection_name mapping (never reconstructed)
  - 300-chunk cap with ingested_at eviction
  - Startup validation of all collections
  - WAL mode on SQLite

Full CRUD and chunk management for P1.2 — bootstrap naming + schema for P0.4.
"""

import hashlib
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import chromadb

from config import (
    MAX_CHUNKS_PER_SESSION,
    EVICTION_BATCH_SIZE,
    SQLITE_DB_PATH,
)

logger = logging.getLogger(__name__)

# ─── ChromaDB persistent client ──────────────────────────────────────────────
_chroma_client = None


def _get_chroma_client():
    """Lazy-init the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.path.join("vectorstore", "chroma_store")
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def _safe_collection_name(raw: str) -> str:
    """
    Sanitise a raw string into a valid ChromaDB collection name.

    Rules:
      - 3–63 chars
      - Alphanumeric + hyphens only
      - Must start and end with alphanumeric
      - No consecutive hyphens
      - Prefixed with 's' to ensure alphanumeric start
      - Hash fallback if sanitised result < 3 chars

    Args:
        raw: Any string (session ID, UUID, etc.)

    Returns:
        A valid ChromaDB collection name.
    """
    # Step 1: Strip to alphanumeric + hyphens only
    cleaned = re.sub(r"[^a-zA-Z0-9\-]", "", raw)

    # Step 2: Collapse consecutive hyphens
    cleaned = re.sub(r"-{2,}", "-", cleaned)

    # Step 3: Strip leading/trailing hyphens
    cleaned = cleaned.strip("-")

    # Step 4: Lowercase
    cleaned = cleaned.lower()

    # Step 5: Prefix with 's' to guarantee alphanumeric start
    cleaned = "s" + cleaned

    # Step 6: Clamp to 63 chars
    cleaned = cleaned[:63]

    # Step 7: Ensure ends with alphanumeric (strip trailing hyphens after clamp)
    cleaned = cleaned.rstrip("-")

    # Step 8: Hash fallback if result < 3 chars
    if len(cleaned) < 3:
        hash_val = hashlib.md5(raw.encode()).hexdigest()[:10]
        cleaned = "s" + hash_val

    return cleaned


class SessionManager:
    """Manages research sessions with ChromaDB collections and SQLite metadata."""

    def __init__(self, db_path: str = None):
        """
        Initialise session manager with SQLite database.

        Args:
            db_path: Path to SQLite database file (default: from config).
        """
        self.db_path = db_path or SQLITE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_db()
        self.chroma = _get_chroma_client()

    def _init_db(self):
        """Create sessions table and enable WAL mode."""
        conn = self._get_conn()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL UNIQUE,
                    topic_label TEXT NOT NULL DEFAULT 'New Research',
                    created_at TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create_session(self) -> str:
        """
        Create a new research session.

        Returns:
            The new session ID (UUID4 string).
        """
        session_id = str(uuid.uuid4())
        collection_name = _safe_collection_name(session_id)
        created_at = datetime.now(timezone.utc).isoformat()

        # Create ChromaDB collection
        self.chroma.get_or_create_collection(name=collection_name)

        # Verify collection was created
        try:
            self.chroma.get_collection(name=collection_name)
        except Exception as e:
            logger.error(f"Failed to verify collection '{collection_name}': {e}")
            raise RuntimeError(f"ChromaDB collection creation failed for '{collection_name}'") from e

        # Store mapping in SQLite
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, collection_name, created_at) VALUES (?, ?, ?)",
                (session_id, collection_name, created_at),
            )
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Created session {session_id} → collection '{collection_name}'")
        return session_id

    def get_session(self, session_id: str) -> dict:
        """
        Look up a session by ID.

        Args:
            session_id: The session UUID.

        Returns:
            Dict with session metadata.

        Raises:
            SessionNotFoundError: If session does not exist.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        return dict(row)

    def get_collection(self, session_id: str):
        """
        Get the ChromaDB collection for a session.

        Args:
            session_id: The session UUID.

        Returns:
            The ChromaDB Collection object.
        """
        session = self.get_session(session_id)
        return self.chroma.get_collection(name=session["collection_name"])

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session and its ChromaDB collection.

        Args:
            session_id: The session UUID.
        """
        session = self.get_session(session_id)
        collection_name = session["collection_name"]

        # Delete ChromaDB collection
        try:
            self.chroma.delete_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Failed to delete collection '{collection_name}': {e}")

        # Delete SQLite record
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

        logger.info(f"Deleted session {session_id}")

    def add_chunks(self, session_id: str, chunks: list[dict]) -> None:
        """
        Add chunks to a session's ChromaDB collection, enforcing the 300-chunk cap.

        Each chunk dict must have: 'id', 'text', 'metadata' (with 'ingested_at').
        Evicts oldest 50 chunks by ingested_at if at capacity.

        Args:
            session_id: The session UUID.
            chunks: List of chunk dicts to add.
        """
        collection = self.get_collection(session_id)
        current_count = collection.count()

        # Evict if at or near capacity
        if current_count + len(chunks) > MAX_CHUNKS_PER_SESSION:
            excess = (current_count + len(chunks)) - MAX_CHUNKS_PER_SESSION
            evict_count = max(excess, EVICTION_BATCH_SIZE)
            self._evict_oldest(collection, evict_count)

        # Add new chunks
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        logger.info(
            f"Added {len(chunks)} chunks to session {session_id}. "
            f"Count: {collection.count()}/{MAX_CHUNKS_PER_SESSION}"
        )

    def _evict_oldest(self, collection, count: int) -> None:
        """
        Evict the oldest chunks by ingested_at metadata.

        Args:
            collection: The ChromaDB collection.
            count: Number of chunks to evict.
        """
        # Get all documents with metadata to find oldest by ingested_at
        all_data = collection.get(include=["metadatas"])
        if not all_data["ids"]:
            return

        # Sort by ingested_at and take the oldest N
        paired = list(zip(all_data["ids"], all_data["metadatas"]))
        paired.sort(key=lambda x: x[1].get("ingested_at", ""))
        to_evict = [p[0] for p in paired[:count]]

        if to_evict:
            collection.delete(ids=to_evict)
            logger.info(f"Evicted {len(to_evict)} oldest chunks from collection")

    def validate_all_collections(self) -> None:
        """
        Startup validation — log orphaned sessions, never crash.

        Checks every session in SQLite has a matching ChromaDB collection.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT id, collection_name FROM sessions").fetchall()
        finally:
            conn.close()

        existing_collections = {c.name for c in self.chroma.list_collections()}

        for row in rows:
            if row["collection_name"] not in existing_collections:
                logger.warning(
                    f"Orphaned session {row['id']}: collection '{row['collection_name']}' "
                    f"not found in ChromaDB. Consider cleanup."
                )

        logger.info(
            f"Validated {len(rows)} sessions against {len(existing_collections)} collections"
        )


class SessionNotFoundError(Exception):
    """Raised when a session_id lookup fails."""
    pass

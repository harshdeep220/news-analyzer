"""
History Store — SQLite-backed message history with async compression.

Responsibilities:
  - append_message(session_id, role, content)
  - get_history(session_id) → list of messages
  - get_compressed_summary(session_id) → max 500-word summary
  - compress_history(session_id) — fires when msg_count > 10,
    summarises oldest 7 into 1 summary block, runs in daemon thread,
    guarded by threading.Lock()
"""

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone

from config import (
    HISTORY_COMPRESS_THRESHOLD,
    HISTORY_MAX_SUMMARY_WORDS,
    FAST_MODEL,
    SQLITE_DB_PATH,
)

logger = logging.getLogger(__name__)

# ─── Thread lock for all SQLite history writes ────────────────────────────────
_history_lock = threading.Lock()


class HistoryStore:
    """SQLite-backed conversation history with async compression."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_DB_PATH
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Create the history table."""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_summary INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_session
                ON message_history(session_id, id)
            """)
            conn.commit()
        finally:
            conn.close()

    def append_message(self, session_id: str, role: str, content: str) -> int:
        """
        Append a message to the session history.

        Args:
            session_id: The session UUID.
            role: 'user' or 'assistant'.
            content: The message content.

        Returns:
            The current message count for this session.
        """
        created_at = datetime.now(timezone.utc).isoformat()

        with _history_lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    "INSERT INTO message_history (session_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, role, content, created_at),
                )
                count = conn.execute(
                    "SELECT COUNT(*) FROM message_history WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                conn.commit()
            finally:
                conn.close()

        return count

    def get_history(self, session_id: str) -> list[dict]:
        """
        Get the full message history for a session.

        Args:
            session_id: The session UUID.

        Returns:
            List of message dicts with role, content, created_at, is_summary.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT role, content, created_at, is_summary "
                "FROM message_history WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def get_message_count(self, session_id: str) -> int:
        """Get the total message count for a session."""
        conn = self._get_conn()
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM message_history WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        finally:
            conn.close()
        return count

    def get_compressed_summary(self, session_id: str) -> str:
        """
        Get a compressed summary of the session history for context passing.
        Max 500 words.

        Args:
            session_id: The session UUID.

        Returns:
            A string summary of the conversation history.
        """
        history = self.get_history(session_id)
        if not history:
            return ""

        # Build a summary from all messages
        parts = []
        for msg in history:
            prefix = "[Summary]" if msg.get("is_summary") else f"[{msg['role']}]"
            parts.append(f"{prefix} {msg['content']}")

        full_text = "\n".join(parts)

        # Truncate to max words
        words = full_text.split()
        if len(words) > HISTORY_MAX_SUMMARY_WORDS:
            full_text = " ".join(words[:HISTORY_MAX_SUMMARY_WORDS]) + "..."

        return full_text

    def trigger_compression_if_needed(self, session_id: str) -> None:
        """
        Check if compression is needed and fire async if so.
        Called AFTER the SSE stream closes — never inline during response.

        Args:
            session_id: The session UUID.
        """
        count = self.get_message_count(session_id)
        if count > HISTORY_COMPRESS_THRESHOLD:
            # Check if we have non-summary messages old enough to compress
            conn = self._get_conn()
            try:
                non_summary_count = conn.execute(
                    "SELECT COUNT(*) FROM message_history "
                    "WHERE session_id = ? AND is_summary = 0",
                    (session_id,),
                ).fetchone()[0]
            finally:
                conn.close()

            if non_summary_count > HISTORY_COMPRESS_THRESHOLD:
                logger.info(
                    f"Triggering async compression for session {session_id} "
                    f"({non_summary_count} non-summary messages)"
                )
                t = threading.Thread(
                    target=self._compress_history,
                    args=(session_id,),
                    daemon=True,
                )
                t.start()

    def _compress_history(self, session_id: str) -> None:
        """
        Compress oldest 7 non-summary messages into 1 summary block.
        Runs in daemon thread, guarded by threading.Lock.

        Args:
            session_id: The session UUID.
        """
        try:
            from infrastructure.google_client import GoogleClient

            with _history_lock:
                conn = self._get_conn()
                try:
                    # Get oldest 7 non-summary messages
                    rows = conn.execute(
                        "SELECT id, role, content FROM message_history "
                        "WHERE session_id = ? AND is_summary = 0 "
                        "ORDER BY id LIMIT 7",
                        (session_id,),
                    ).fetchall()

                    if len(rows) < 7:
                        return

                    # Build text to summarize
                    messages_text = "\n".join(
                        f"[{row['role']}]: {row['content']}" for row in rows
                    )

                    # Use Flash to summarize
                    client = GoogleClient()
                    summary = client.generate(
                        prompt=(
                            f"Summarize this conversation concisely in under 150 words. "
                            f"Capture key topics, questions asked, and conclusions reached:\n\n"
                            f"{messages_text}"
                        ),
                        model=FAST_MODEL,
                        system="You are a conversation summarizer. Be concise and factual.",
                    )

                    # Delete the 7 old messages and insert the summary
                    ids_to_delete = [row["id"] for row in rows]
                    placeholders = ",".join("?" * len(ids_to_delete))
                    conn.execute(
                        f"DELETE FROM message_history WHERE id IN ({placeholders})",
                        ids_to_delete,
                    )
                    conn.execute(
                        "INSERT INTO message_history "
                        "(session_id, role, content, created_at, is_summary) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            session_id,
                            "system",
                            summary,
                            datetime.now(timezone.utc).isoformat(),
                            1,
                        ),
                    )
                    conn.commit()
                    logger.info(
                        f"Compressed 7 messages into summary for session {session_id}"
                    )
                finally:
                    conn.close()

        except Exception as e:
            logger.error(f"History compression failed for session {session_id}: {e}")

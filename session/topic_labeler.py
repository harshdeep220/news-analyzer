"""
Topic Labeler — Auto-labels sessions when conversation content is sufficient.

Rules:
  - Fires only when total word count across session exceeds 100 words
  - AND label is still 'New Research'
  - Never re-triggers once label is set
  - Uses Flash for topic extraction
"""

import logging
import sqlite3
import os

from config import TOPIC_LABEL_WORD_THRESHOLD, FAST_MODEL, SQLITE_DB_PATH

logger = logging.getLogger(__name__)


class TopicLabeler:
    """Auto-labels research sessions based on conversation content."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_DB_PATH

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def maybe_label(self, session_id: str, history: list[dict]) -> str | None:
        """
        Check if the session should be labeled and label it if conditions are met.

        Args:
            session_id: The session UUID.
            history: List of message dicts from HistoryStore.

        Returns:
            The new label if updated, None if no update needed.
        """
        # Check current label
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT topic_label FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None

        current_label = row["topic_label"]

        # Never re-trigger once label is set
        if current_label != "New Research":
            return None

        # Count total words across all messages
        total_words = sum(
            len(msg.get("content", "").split())
            for msg in history
        )

        if total_words < TOPIC_LABEL_WORD_THRESHOLD:
            return None

        # Generate topic label using Flash
        try:
            from infrastructure.google_client import GoogleClient
            client = GoogleClient()

            # Build conversation text for labeling
            conv_text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in history[-5:]  # Last 5 messages for context
            )

            label = client.generate(
                prompt=(
                    f"Generate a short topic label (3-6 words) for this research conversation. "
                    f"Return ONLY the label, nothing else:\n\n{conv_text}"
                ),
                model=FAST_MODEL,
                system="You generate concise topic labels for research sessions.",
            )

            label = label.strip().strip('"').strip("'")[:50]  # Clean and cap

            if label:
                conn = self._get_conn()
                try:
                    conn.execute(
                        "UPDATE sessions SET topic_label = ? WHERE id = ? AND topic_label = 'New Research'",
                        (label, session_id),
                    )
                    conn.commit()
                finally:
                    conn.close()

                logger.info(f"Session {session_id} labeled: '{label}'")
                return label

        except Exception as e:
            logger.error(f"Topic labeling failed for session {session_id}: {e}")

        return None

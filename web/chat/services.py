"""
Service layer — bridges Django ORM, ChromaDB, and the orchestrator.

Drop-in replacements for the old SessionManager and HistoryStore,
backed by Django models instead of raw SQLite.
"""

import hashlib
import json
import logging
import os
import re
import threading
import uuid

import chromadb
from django.utils import timezone

from chat.models import Session, Message, CredibilityCache
from config import (
    MAX_CHUNKS_PER_SESSION,
    EVICTION_BATCH_SIZE,
    HISTORY_COMPRESS_THRESHOLD,
    HISTORY_MAX_SUMMARY_WORDS,
    FAST_MODEL,
    TOPIC_LABEL_WORD_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ─── ChromaDB Client ─────────────────────────────────────────────────────────
_chroma_client = None


def _get_chroma_client():
    """Lazy-init the ChromaDB persistent client."""
    global _chroma_client
    if _chroma_client is None:
        # Resolve relative to project root
        from django.conf import settings
        persist_dir = os.path.join(str(settings.PROJECT_ROOT), "vectorstore", "chroma_store")
        os.makedirs(persist_dir, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def _safe_collection_name(raw: str) -> str:
    """Sanitise a raw string into a valid ChromaDB collection name (3-63 chars)."""
    cleaned = re.sub(r"[^a-zA-Z0-9\-]", "", raw)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-").lower()
    cleaned = "s" + cleaned
    cleaned = cleaned[:63].rstrip("-")
    if len(cleaned) < 3:
        hash_val = hashlib.md5(raw.encode()).hexdigest()[:10]
        cleaned = "s" + hash_val
    return cleaned


# ─── SessionManager (Django ORM) ─────────────────────────────────────────────

class SessionManager:
    """
    Drop-in replacement for session.session_manager.SessionManager.
    Backed by Django ORM + ChromaDB.
    """

    def __init__(self):
        self.chroma = _get_chroma_client()

    def create_session(self) -> str:
        """Create a new session with ChromaDB collection. Returns session_id string."""
        session_id = uuid.uuid4()
        collection_name = _safe_collection_name(str(session_id))

        # Create ChromaDB collection
        self.chroma.get_or_create_collection(name=collection_name)

        # Create Django model
        Session.objects.create(
            id=session_id,
            collection_name=collection_name,
        )

        logger.info(f"Created session {session_id} → collection '{collection_name}'")
        return str(session_id)

    def get_session(self, session_id: str) -> dict:
        """Look up session by ID. Returns dict compatible with old interface."""
        try:
            s = Session.objects.get(id=session_id)
            return {
                "id": str(s.id),
                "collection_name": s.collection_name,
                "topic_label": s.topic_label,
                "created_at": s.created_at.isoformat(),
                "message_count": s.message_count,
            }
        except Session.DoesNotExist:
            raise SessionNotFoundError(f"Session '{session_id}' not found")

    def get_collection(self, session_id: str):
        """Get the ChromaDB collection for a session."""
        session = self.get_session(session_id)
        return self.chroma.get_collection(name=session["collection_name"])

    def delete_session(self, session_id: str) -> None:
        """Delete session, its messages, and ChromaDB collection."""
        try:
            session_data = self.get_session(session_id)
            collection_name = session_data["collection_name"]

            # Delete ChromaDB collection
            try:
                self.chroma.delete_collection(name=collection_name)
            except Exception as e:
                logger.warning(f"Failed to delete collection '{collection_name}': {e}")

            # Django cascade deletes messages automatically
            Session.objects.filter(id=session_id).delete()
            logger.info(f"Deleted session {session_id}")
        except SessionNotFoundError:
            pass

    def add_chunks(self, session_id: str, chunks: list[dict]) -> None:
        """Add chunks to session's ChromaDB collection with 300-chunk cap."""
        collection = self.get_collection(session_id)
        current_count = collection.count()

        if current_count + len(chunks) > MAX_CHUNKS_PER_SESSION:
            excess = (current_count + len(chunks)) - MAX_CHUNKS_PER_SESSION
            evict_count = max(excess, EVICTION_BATCH_SIZE)
            self._evict_oldest(collection, evict_count)

        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def _evict_oldest(self, collection, count: int) -> None:
        """Evict oldest chunks by ingested_at metadata."""
        all_data = collection.get(include=["metadatas"])
        if not all_data["ids"]:
            return
        paired = list(zip(all_data["ids"], all_data["metadatas"]))
        paired.sort(key=lambda x: x[1].get("ingested_at", ""))
        to_evict = [p[0] for p in paired[:count]]
        if to_evict:
            collection.delete(ids=to_evict)

    def list_sessions(self) -> list[dict]:
        """List all sessions ordered by most recent."""
        sessions = Session.objects.all()
        return [
            {
                "id": str(s.id),
                "topic_label": s.topic_label,
                "created_at": s.created_at.isoformat(),
                "message_count": s.messages.count(),
            }
            for s in sessions
        ]


# ─── HistoryStore (Django ORM) ────────────────────────────────────────────────

class HistoryStore:
    """
    Drop-in replacement for session.history_store.HistoryStore.
    Backed by Django ORM.
    """

    def append_message(self, session_id: str, role: str, content: str) -> int:
        """Append message and return current count."""
        Message.objects.create(
            session_id=session_id,
            role=role,
            content=content,
        )
        count = Message.objects.filter(session_id=session_id).count()

        # Update session message count
        Session.objects.filter(id=session_id).update(message_count=count)

        return count

    def get_history(self, session_id: str) -> list[dict]:
        """Get full message history for a session."""
        messages = Message.objects.filter(session_id=session_id).order_by("id")
        return [
            {
                "role": m.role,
                "content": m.content,
                "created_at": m.created_at.isoformat(),
                "is_summary": m.is_summary,
            }
            for m in messages
        ]

    def get_message_count(self, session_id: str) -> int:
        """Get message count for a session."""
        return Message.objects.filter(session_id=session_id).count()

    def get_compressed_summary(self, session_id: str) -> str:
        """Get compressed summary of session history (max 500 words)."""
        history = self.get_history(session_id)
        if not history:
            return ""

        parts = []
        for msg in history:
            prefix = "[Summary]" if msg.get("is_summary") else f"[{msg['role']}]"
            parts.append(f"{prefix} {msg['content']}")

        full_text = "\n".join(parts)
        words = full_text.split()
        if len(words) > HISTORY_MAX_SUMMARY_WORDS:
            full_text = " ".join(words[:HISTORY_MAX_SUMMARY_WORDS]) + "..."

        return full_text

    def trigger_compression_if_needed(self, session_id: str) -> None:
        """Fire async compression if message count exceeds threshold."""
        non_summary_count = Message.objects.filter(
            session_id=session_id, is_summary=False
        ).count()

        if non_summary_count > HISTORY_COMPRESS_THRESHOLD:
            logger.info(f"Triggering compression for session {session_id}")
            t = threading.Thread(
                target=self._compress_history,
                args=(session_id,),
                daemon=True,
            )
            t.start()

    def _compress_history(self, session_id: str) -> None:
        """Compress oldest 7 non-summary messages into 1 summary block."""
        try:
            from infrastructure.google_client import GoogleClient

            oldest = Message.objects.filter(
                session_id=session_id, is_summary=False
            ).order_by("id")[:7]

            if len(oldest) < 7:
                return

            messages_text = "\n".join(
                f"[{m.role}]: {m.content}" for m in oldest
            )

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

            # Delete old messages and insert summary
            ids_to_delete = [m.id for m in oldest]
            Message.objects.filter(id__in=ids_to_delete).delete()
            Message.objects.create(
                session_id=session_id,
                role="system",
                content=summary,
                is_summary=True,
            )
            logger.info(f"Compressed 7 messages for session {session_id}")

        except Exception as e:
            logger.error(f"Compression failed for {session_id}: {e}")


# ─── TopicLabeler (Django ORM) ────────────────────────────────────────────────

class TopicLabeler:
    """
    Auto-labels sessions based on conversation content.
    Replaces session.topic_labeler.TopicLabeler.
    """

    def maybe_label(self, session_id: str, history: list[dict]) -> str | None:
        """Label session if conditions met. Returns new label or None."""
        try:
            session = Session.objects.get(id=session_id)
        except Session.DoesNotExist:
            return None

        if session.topic_label != "New Research":
            return None

        total_words = sum(len(msg.get("content", "").split()) for msg in history)
        if total_words < TOPIC_LABEL_WORD_THRESHOLD:
            return None

        try:
            from infrastructure.google_client import GoogleClient
            client = GoogleClient()

            conv_text = "\n".join(
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in history[-5:]
            )

            label = client.generate(
                prompt=(
                    f"Generate a short topic label (3-6 words) for this research conversation. "
                    f"Return ONLY the label, nothing else:\n\n{conv_text}"
                ),
                model=FAST_MODEL,
                system="You generate concise topic labels for research sessions.",
            )

            label = label.strip().strip('"').strip("'")[:50]
            if label:
                Session.objects.filter(
                    id=session_id, topic_label="New Research"
                ).update(topic_label=label)
                logger.info(f"Session {session_id} labeled: '{label}'")
                return label

        except Exception as e:
            logger.error(f"Topic labeling failed: {e}")

        return None


# ─── Exception ────────────────────────────────────────────────────────────────

class SessionNotFoundError(Exception):
    """Raised when a session_id lookup fails."""
    pass

"""
Django ORM models — replaces raw SQLite session_manager.py and history_store.py.

Models:
  - Session: research session with topic label and ChromaDB collection mapping
  - Message: conversation message (user/assistant/system) with compression support
  - CredibilityCache: URL-keyed credibility score cache
"""

import uuid
from django.db import models
from django.utils import timezone


class Session(models.Model):
    """
    Research session — replaces the raw SQLite `sessions` table.

    Each session maps to a ChromaDB collection for vector storage.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    collection_name = models.CharField(max_length=63, unique=True)
    topic_label = models.CharField(max_length=100, default="New Research")
    created_at = models.DateTimeField(default=timezone.now)
    message_count = models.IntegerField(default=0)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.topic_label} ({self.id})"


class Message(models.Model):
    """
    Conversation message — replaces the raw SQLite `message_history` table.

    Supports summary compression: is_summary=True marks compressed blocks.
    """
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
    ]

    session = models.ForeignKey(
        Session, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    is_summary = models.BooleanField(default=False)

    class Meta:
        ordering = ["id"]
        indexes = [
            models.Index(fields=["session", "id"]),
        ]

    def __str__(self):
        return f"[{self.role}] {self.content[:50]}..."


class CredibilityCache(models.Model):
    """
    URL-keyed credibility score cache — replaces the raw SQLite `credibility_cache` table.
    """
    url = models.URLField(primary_key=True, max_length=2048)
    total = models.IntegerField()
    signals_json = models.JSONField()
    scored_at = models.DateTimeField(default=timezone.now)

    class Meta:
        verbose_name_plural = "Credibility cache entries"

    def __str__(self):
        return f"{self.url[:60]} → {self.total}/100"

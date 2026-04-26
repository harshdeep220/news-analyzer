"""
Query Cache — In-memory cache to prevent redundant pipeline executions.

Key = sha256(session_id + normalized_query), TTL = 60 seconds.
Check before LangGraph entry — cache hit means zero API calls.

Full implementation is P1.10. This is the bootstrap stub with the class
signature so imports work during Phase 0.
"""

import hashlib
import time
import logging

from config import QUERY_CACHE_TTL

logger = logging.getLogger(__name__)


class QueryCache:
    """In-memory query response cache with TTL-based expiration."""

    def __init__(self, ttl: int = None):
        self._cache: dict[str, tuple[object, float]] = {}
        self._ttl = ttl or QUERY_CACHE_TTL

    def _make_key(self, session_id: str, query: str) -> str:
        """Generate cache key from session_id + normalized query."""
        normalized = (session_id + query.strip().lower()).encode()
        return hashlib.sha256(normalized).hexdigest()

    def get(self, session_id: str, query: str) -> object | None:
        """
        Look up cached response. Returns None on miss or expired entry.
        Evicts expired entries on each call.
        """
        key = self._make_key(session_id, query)
        self._evict_expired()

        entry = self._cache.get(key)
        if entry is None:
            return None

        response, stored_at = entry
        if time.time() - stored_at > self._ttl:
            del self._cache[key]
            return None

        logger.info(f"QueryCache HIT for key {key[:12]}...")
        return response

    def store(self, session_id: str, query: str, response: object) -> None:
        """Store a response in the cache."""
        key = self._make_key(session_id, query)
        self._cache[key] = (response, time.time())
        logger.info(f"QueryCache STORE for key {key[:12]}...")

    def _evict_expired(self) -> None:
        """Remove all expired entries from the cache."""
        now = time.time()
        expired_keys = [
            k for k, (_, stored_at) in self._cache.items()
            if now - stored_at > self._ttl
        ]
        for k in expired_keys:
            del self._cache[k]

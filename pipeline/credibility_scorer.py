"""
Credibility Scorer — 4-Signal pipeline.

Signals:
  1. source_tier: lookup from source_credibility_db.json (50+ domains)
  2. corroboration: count independent sources reporting same claims
  3. content_quality: Flash-graded writing quality (0-100)
  4. freshness: recency scoring based on published_date

URL-keyed SQLite cache — never re-score same URL.
Corroboration: 0 independent → 10, 1 → 30, 2-3 → 60, 4+ → 85.
Wire-republisher domains weighted at 0.5x.
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from infrastructure.google_client import GoogleClient
from config import (
    FAST_MODEL,
    SQLITE_DB_PATH,
    CRED_WEIGHT_SOURCE_TIER,
    CRED_WEIGHT_CORROBORATION,
    CRED_WEIGHT_CONTENT_QUALITY,
    CRED_WEIGHT_FRESHNESS,
)

logger = logging.getLogger(__name__)

# ─── Load source credibility DB ──────────────────────────────────────────────
_CRED_DB_PATH = os.path.join("data", "source_credibility_db.json")
_CRED_DB: dict = {}


def _load_cred_db():
    """Load or generate the source credibility database."""
    global _CRED_DB
    if _CRED_DB:
        return _CRED_DB

    if os.path.exists(_CRED_DB_PATH):
        with open(_CRED_DB_PATH, "r", encoding="utf-8") as f:
            _CRED_DB = json.load(f)
            logger.info(f"Loaded credibility DB: {len(_CRED_DB)} domains")
            return _CRED_DB

    # Generate with Flash if missing
    logger.info("Generating source_credibility_db.json with Flash...")
    try:
        client = GoogleClient()
        response = client.generate(
            prompt=(
                "Generate a JSON object mapping 60+ news outlet domains to their credibility tier (1-4). "
                "Tier 1 = most credible (score 90-100), Tier 2 = reliable (70-89), "
                "Tier 3 = mixed reliability (40-69), Tier 4 = low reliability (10-39). "
                "Include domains for: major wire services (AP, Reuters, AFP), "
                "major US papers (NYT, WSJ, WaPo), major UK papers (BBC, Guardian, FT), "
                "major Indian outlets (NDTV, Hindu, TOI, HindustanTimes), "
                "tech sites (TechCrunch, Verge, Wired, ArsTechnica), "
                "financial (Bloomberg, CNBC, MarketWatch), "
                "international (AlJazeera, DW, France24, NHK), "
                "aggregators and blogs (Medium, Substack, WordPress sites). "
                "Format: {\"domain.com\": {\"tier\": 1, \"score\": 95, \"type\": \"wire_service|newspaper|broadcaster|digital_native|blog|aggregator\"}} "
                "Return ONLY valid JSON, no other text."
            ),
            model=FAST_MODEL,
            system="You are a media credibility expert. Generate accurate domain credibility data.",
        )

        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        _CRED_DB = json.loads(text)
        os.makedirs(os.path.dirname(_CRED_DB_PATH), exist_ok=True)
        with open(_CRED_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(_CRED_DB, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated credibility DB: {len(_CRED_DB)} domains")

    except Exception as e:
        logger.error(f"Failed to generate credibility DB: {e}")
        _CRED_DB = {}

    return _CRED_DB


# ─── Wire republisher detection ──────────────────────────────────────────────
WIRE_REPUBLISHER_INDICATORS = [
    "yahoo.com", "msn.com", "news.google.com",
    "inshorts.com", "smartnews.com",
]


@dataclass
class CredibilityScore:
    """4-signal credibility score."""
    total: int = 0
    signals: dict = field(default_factory=lambda: {
        "source_tier": 0,
        "corroboration": 0,
        "content_quality": 0,
        "freshness": 0,
    })
    url: str = ""
    cached: bool = False


class CredibilityScorer:
    """4-signal credibility scorer with SQLite caching."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or SQLITE_DB_PATH
        self._client = GoogleClient()
        self._cred_db = _load_cred_db()
        self._init_cache_table()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_cache_table(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credibility_cache (
                    url TEXT PRIMARY KEY,
                    total INTEGER NOT NULL,
                    signals_json TEXT NOT NULL,
                    scored_at TEXT NOT NULL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def score(self, article: dict, batch_articles: list[dict] = None) -> CredibilityScore:
        """
        Score a single article's credibility.

        Args:
            article: Dict with url, content, outlet, published_date.
            batch_articles: Other articles in the batch (for corroboration).

        Returns:
            CredibilityScore with 4-signal breakdown.
        """
        url = article.get("url", "")

        # Check cache first
        cached = self._get_cached(url)
        if cached:
            return cached

        # 1. Source tier
        source_score = self._score_source_tier(article)

        # 2. Corroboration
        corroboration_score = self._score_corroboration(article, batch_articles or [])

        # 3. Content quality
        content_score = self._score_content_quality(article)

        # 4. Freshness
        freshness_score = self._score_freshness(article)

        # Weighted total
        total = int(
            source_score * CRED_WEIGHT_SOURCE_TIER
            + corroboration_score * CRED_WEIGHT_CORROBORATION
            + content_score * CRED_WEIGHT_CONTENT_QUALITY
            + freshness_score * CRED_WEIGHT_FRESHNESS
        )

        result = CredibilityScore(
            total=max(0, min(100, total)),
            signals={
                "source_tier": source_score,
                "corroboration": corroboration_score,
                "content_quality": content_score,
                "freshness": freshness_score,
            },
            url=url,
        )

        # Cache result
        self._cache_result(result)

        return result

    def _score_source_tier(self, article: dict) -> int:
        """Score based on source credibility database."""
        url = article.get("url", "")
        outlet = article.get("outlet", "")

        # Extract domain
        domain = ""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
        except Exception:
            domain = outlet

        # Lookup in DB
        if domain in self._cred_db:
            entry = self._cred_db[domain]
            return entry.get("score", 50)

        # Check partial matches
        for db_domain, entry in self._cred_db.items():
            if db_domain in domain or domain in db_domain:
                return entry.get("score", 50)

        # Unknown source — neutral, not negative
        return 50

    def _score_corroboration(self, article: dict, batch_articles: list[dict]) -> int:
        """Score based on how many other sources report same claims."""
        if not batch_articles or len(batch_articles) < 2:
            return 45  # Can't corroborate — neutral, not a penalty

        url = article.get("url", "")
        title = article.get("title", "").lower()
        content_start = article.get("content", "")[:300].lower()

        # Simple keyword overlap for corroboration
        match_count = 0
        title_words = set(w for w in title.split() if len(w) > 4)

        for other in batch_articles:
            if other.get("url", "") == url:
                continue

            other_title = other.get("title", "").lower()
            other_content = other.get("content", "")[:300].lower()
            other_words = set(w for w in other_title.split() if len(w) > 4)

            # Check keyword overlap
            shared = title_words & other_words
            if len(shared) >= 2:
                # Check if wire republisher
                other_domain = other.get("outlet", "")
                if any(wr in other_domain for wr in WIRE_REPUBLISHER_INDICATORS):
                    match_count += 0.5  # Wire republisher weighted at 0.5x
                else:
                    match_count += 1

        # Scoring: 0→45, 1→55, 2-3→70, 4+→85
        if match_count == 0:
            return 45
        elif match_count < 2:
            return 55
        elif match_count < 4:
            return 60
        else:
            return 85

    def _score_content_quality(self, article: dict) -> int:
        """Score content quality using text heuristics — zero API calls."""
        content = article.get("content", "")
        if len(content) < 200:
            return 35

        score = 55  # Base score

        # Attribution signals (quotes, "said", "according to")
        import re
        quote_count = content.count('"') // 2
        attribution_words = len(re.findall(
            r'\b(said|according to|reported|confirmed|announced|stated|told)\b',
            content, re.IGNORECASE
        ))
        if attribution_words >= 3:
            score += 15
        elif attribution_words >= 1:
            score += 8

        if quote_count >= 2:
            score += 10
        elif quote_count >= 1:
            score += 5

        # Factual specificity (numbers, dates, percentages)
        numbers = len(re.findall(r'\b\d+[\.,]?\d*%?\b', content[:2000]))
        if numbers >= 5:
            score += 10
        elif numbers >= 2:
            score += 5

        # Content length (longer = more thorough)
        word_count = len(content.split())
        if word_count >= 800:
            score += 10
        elif word_count >= 400:
            score += 5

        # Proper nouns density (names, places = specificity)
        proper_nouns = len(re.findall(r'\b[A-Z][a-z]{2,}\b', content[:1500]))
        if proper_nouns >= 15:
            score += 5

        return min(100, max(0, score))

    def _score_freshness(self, article: dict) -> int:
        """Score based on publication recency."""
        pub_date = article.get("published_date", "")
        if not pub_date:
            return 50

        try:
            pub_dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = (now - pub_dt).total_seconds() / 3600

            if age_hours < 1:
                return 100
            elif age_hours < 6:
                return 90
            elif age_hours < 24:
                return 75
            elif age_hours < 72:
                return 50
            elif age_hours < 168:  # 1 week
                return 30
            else:
                return 10
        except Exception:
            return 50

    def _get_cached(self, url: str) -> CredibilityScore | None:
        """Check SQLite cache for URL."""
        if not url:
            return None
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT total, signals_json FROM credibility_cache WHERE url = ?",
                (url,),
            ).fetchone()
            if row:
                return CredibilityScore(
                    total=row["total"],
                    signals=json.loads(row["signals_json"]),
                    url=url,
                    cached=True,
                )
        finally:
            conn.close()
        return None

    def _cache_result(self, score: CredibilityScore):
        """Store result in SQLite cache."""
        if not score.url:
            return
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO credibility_cache (url, total, signals_json, scored_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    score.url,
                    score.total,
                    json.dumps(score.signals),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

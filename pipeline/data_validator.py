"""
Data Validator — Validates Tavily articles before they touch ChromaDB.

Every article passes through validate() before embedding. Checks:
  - has_url
  - content_length > 200 (and < 50000)
  - not_paywalled
  - not_empty_scrape
  - is_wire_duplicate (MD5 of first 300 chars, lowercased)
  - published_date <= today

Batch method: validate_batch(articles, seen_hashes=None)
  - seen_hashes is caller-controlled, per-batch scope
  - Never raises exception
"""

import hashlib
import logging
import re
from datetime import datetime, timezone

from config import MIN_CONTENT_LENGTH, MAX_CONTENT_LENGTH, WIRE_DEDUP_CHARS

logger = logging.getLogger(__name__)

# ─── Paywall patterns ─────────────────────────────────────────────────────────
PAYWALL_PATTERNS = [
    r"subscribe to (continue|read|access)",
    r"this (content|article) is (for|available to) (subscribers|members|premium)",
    r"you('ve| have) reached your (free|monthly) (article|limit)",
    r"sign in to (read|continue|access)",
    r"create (a |an )?(free )?account to continue",
    r"already a subscriber\?",
    r"unlock this (article|story)",
    r"to continue reading",
    r"register for free",
]

PAYWALL_RE = re.compile("|".join(PAYWALL_PATTERNS), re.IGNORECASE)

# ─── Empty scrape indicators ─────────────────────────────────────────────────
EMPTY_SCRAPE_PATTERNS = [
    "javascript is required",
    "enable javascript",
    "please enable cookies",
    "access denied",
    "403 forbidden",
    "page not found",
    "404 error",
    "captcha",
]


def _is_paywalled(content: str) -> bool:
    """Check if content appears to be paywalled."""
    return bool(PAYWALL_RE.search(content[:1000]))


def _is_empty_scrape(content: str) -> bool:
    """Check if content is an empty or failed scrape."""
    content_lower = content.lower().strip()
    if len(content_lower) < 50:
        return True
    for pattern in EMPTY_SCRAPE_PATTERNS:
        if pattern in content_lower[:500]:
            return True
    return False


def _wire_hash(content: str) -> str:
    """Generate MD5 hash of first 300 chars (lowercased, stripped) for wire dedup."""
    normalized = content[:WIRE_DEDUP_CHARS].lower().strip()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


class DataValidator:
    """Validates articles from Tavily before they enter the pipeline."""

    @staticmethod
    def validate(article: dict, seen_hashes: set = None) -> tuple[bool, dict]:
        """
        Validate a single article.

        Args:
            article: Dict with keys: url, content, published_date (optional).
            seen_hashes: Set of MD5 hashes for wire-story dedup (per-batch scope).

        Returns:
            Tuple of (is_valid: bool, details: dict with rejection reasons or cleaned article).
        """
        reasons = []

        # ─── URL check ────────────────────────────────────────────────
        url = article.get("url", "")
        if not url or not url.startswith("http"):
            reasons.append("missing_or_invalid_url")

        # ─── Content extraction ───────────────────────────────────────
        content = article.get("content", "") or article.get("raw_content", "") or ""

        # ─── Content length ───────────────────────────────────────────
        if len(content) < MIN_CONTENT_LENGTH:
            reasons.append(f"content_too_short ({len(content)} < {MIN_CONTENT_LENGTH})")

        if len(content) > MAX_CONTENT_LENGTH:
            reasons.append(f"content_too_long ({len(content)} > {MAX_CONTENT_LENGTH})")

        # ─── Paywall check ────────────────────────────────────────────
        if content and _is_paywalled(content):
            reasons.append("paywalled")

        # ─── Empty scrape check ───────────────────────────────────────
        if content and _is_empty_scrape(content):
            reasons.append("empty_scrape")

        # ─── Wire duplicate check ─────────────────────────────────────
        if content and seen_hashes is not None:
            content_hash = _wire_hash(content)
            if content_hash in seen_hashes:
                reasons.append("wire_duplicate")
            else:
                seen_hashes.add(content_hash)

        # ─── Date cap ────────────────────────────────────────────────
        metadata_warning = None
        published_date = article.get("published_date", "")
        if published_date:
            try:
                pub_dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                today = datetime.now(timezone.utc)
                if pub_dt > today:
                    published_date = today.isoformat()
                    metadata_warning = f"future_date_capped (original: {article.get('published_date')})"
                    logger.warning(f"Future date capped for {url}: {metadata_warning}")
            except (ValueError, TypeError):
                pass  # Unparseable date — leave as-is

        if reasons:
            logger.debug(f"Article rejected [{url[:60]}]: {reasons}")
            return False, {"url": url, "reasons": reasons}

        # ─── Return cleaned article ───────────────────────────────────
        return True, {
            "url": url,
            "content": content,
            "title": article.get("title", ""),
            "published_date": published_date,
            "metadata_warning": metadata_warning,
            "score": article.get("score", 0.0),
        }

    @staticmethod
    def validate_batch(
        articles: list[dict], seen_hashes: set = None
    ) -> tuple[list[dict], list[dict]]:
        """
        Validate a batch of articles.

        Args:
            articles: List of article dicts from Tavily.
            seen_hashes: Caller-controlled set for wire dedup. Per-batch scope — NOT global.

        Returns:
            Tuple of (valid_articles, rejected_articles).
        """
        if seen_hashes is None:
            seen_hashes = set()

        valid = []
        rejected = []

        for article in articles:
            try:
                is_valid, details = DataValidator.validate(article, seen_hashes)
                if is_valid:
                    valid.append(details)
                else:
                    rejected.append(details)
            except Exception as e:
                logger.error(f"Validation error for article: {e}")
                rejected.append({
                    "url": article.get("url", "unknown"),
                    "reasons": [f"validation_error: {str(e)}"],
                })

        logger.info(
            f"Validated batch: {len(valid)} accepted, {len(rejected)} rejected "
            f"out of {len(articles)} total"
        )
        return valid, rejected

"""
Source Comparator — Compares articles covering the same story.

Same-story trigger: extract top-5 named entities from each title using
regex + stopword filter (no API call). If ≥3 shared entities AND published
within 24h → fire comparison.

First 800 words per article passed to comparator.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from infrastructure.google_client import GoogleClient
from config import (
    FAST_MODEL,
    COMPARATOR_ENTITY_THRESHOLD,
    COMPARATOR_TIME_WINDOW_HOURS,
    COMPARATOR_MAX_WORDS,
)

logger = logging.getLogger(__name__)

# Stopwords for entity extraction (no API call)
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "and", "but", "or", "if", "up", "about",
    "says", "said", "new", "also", "just", "us", "it", "he", "she", "they",
    "i", "we", "you", "my", "his", "her", "its", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "while",
    "report", "reports", "news", "breaking", "update", "latest", "today",
}


@dataclass
class ComparisonResult:
    """Result of source comparison."""
    agreed_facts: list[str] = field(default_factory=list)
    disputed_facts: list[str] = field(default_factory=list)
    unique_claims: dict = field(default_factory=dict)  # outlet → list[str]
    framing_differences: str = ""
    articles_compared: int = 0
    triggered: bool = False


def extract_entities(title: str) -> set[str]:
    """
    Extract named entity candidates from a title via regex + stopword filter.
    No API call — pure regex.

    Args:
        title: Article title.

    Returns:
        Set of potential entity strings (capitalized words, proper nouns).
    """
    # Extract words that start with capital letters (likely proper nouns)
    words = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', title)

    # Also extract all words > 3 chars for general keyword matching
    all_words = set(
        w.lower() for w in title.split()
        if len(w) > 3 and w.lower() not in STOPWORDS
    )

    # Combine capitalized words (entities) with significant keywords
    entities = set(w.lower() for w in words if w.lower() not in STOPWORDS)
    entities.update(all_words)

    return entities


class SourceComparator:
    """Compares multiple articles covering the same story."""

    def __init__(self):
        self._client = GoogleClient()

    def should_compare(self, articles: list[dict]) -> list[tuple[int, int]]:
        """
        Determine which article pairs should be compared.
        Uses entity overlap + time window — zero API calls.

        Args:
            articles: List of article dicts.

        Returns:
            List of (index_a, index_b) pairs that triggered comparison.
        """
        pairs = []
        if len(articles) < 2:
            return pairs

        # Extract entities for each article
        entities_per_article = []
        dates_per_article = []

        for article in articles:
            title = article.get("title", "")
            entities = extract_entities(title)
            entities_per_article.append(entities)

            # Parse date
            pub_date = article.get("published_date", "")
            try:
                dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                dt = None
            dates_per_article.append(dt)

        # Check all pairs
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                shared = entities_per_article[i] & entities_per_article[j]

                if len(shared) >= COMPARATOR_ENTITY_THRESHOLD:
                    # Check time window
                    dt_i, dt_j = dates_per_article[i], dates_per_article[j]
                    if dt_i and dt_j:
                        delta = abs((dt_i - dt_j).total_seconds()) / 3600
                        if delta <= COMPARATOR_TIME_WINDOW_HOURS:
                            pairs.append((i, j))
                            logger.info(
                                f"Same-story trigger: {len(shared)} shared entities "
                                f"({', '.join(list(shared)[:5])}), "
                                f"{delta:.1f}h apart"
                            )
                    else:
                        # No date info — compare anyway if entity overlap is strong
                        if len(shared) >= COMPARATOR_ENTITY_THRESHOLD + 1:
                            pairs.append((i, j))

        return pairs

    def compare(self, articles: list[dict]) -> ComparisonResult:
        """
        Compare 2+ articles about the same story.

        Args:
            articles: List of article dicts to compare.

        Returns:
            ComparisonResult with agreed/disputed facts and framing differences.
        """
        if len(articles) < 2:
            return ComparisonResult(triggered=False)

        # Cap at first 800 words per article
        article_texts = []
        for i, article in enumerate(articles):
            content = article.get("content", "")
            words = content.split()[:COMPARATOR_MAX_WORDS]
            truncated = " ".join(words)
            outlet = article.get("outlet", f"Source {i+1}")
            article_texts.append(f"[{outlet}]:\n{truncated}")

        combined = "\n\n---\n\n".join(article_texts)

        try:
            response = self._client.generate(
                prompt=(
                    f"Compare these {len(articles)} articles about the same event.\n\n"
                    f"{combined}\n\n"
                    f"Return ONLY this JSON:\n"
                    f'{{\n'
                    f'  "agreed_facts": ["fact1", "fact2", "fact3"],\n'
                    f'  "disputed_facts": ["fact where sources disagree"],\n'
                    f'  "unique_claims": {{"outlet_name": ["claim only this source makes"]}},\n'
                    f'  "framing_differences": "Brief description of how sources frame the story differently"\n'
                    f'}}\n\n'
                    f"Note: Comparison based on first {COMPARATOR_MAX_WORDS} words of each article.\n"
                    f"Return ONLY valid JSON."
                ),
                model=FAST_MODEL,
                system=(
                    "You are a media comparison analyst. Compare source coverage objectively. "
                    "Identify what sources agree on, disagree on, and how they frame stories differently."
                ),
            )

            return self._parse_result(response, len(articles))

        except Exception as e:
            logger.error(f"Source comparison failed: {e}")
            return ComparisonResult(triggered=True, articles_compared=len(articles))

    def _parse_result(self, response: str, article_count: int) -> ComparisonResult:
        """Parse LLM response into ComparisonResult."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            return ComparisonResult(
                agreed_facts=parsed.get("agreed_facts", []),
                disputed_facts=parsed.get("disputed_facts", []),
                unique_claims=parsed.get("unique_claims", {}),
                framing_differences=parsed.get("framing_differences", ""),
                articles_compared=article_count,
                triggered=True,
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse comparison result: {e}")
            return ComparisonResult(triggered=True, articles_compared=article_count)

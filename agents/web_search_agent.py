"""
Web Search Agent — Tavily wrapper with dedup and date capping.

Searches Tavily for live news, validates results through DataValidator,
and returns clean Article dataclasses ready for embedding.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tavily import TavilyClient

from config import TAVILY_API_KEY
from pipeline.data_validator import DataValidator

logger = logging.getLogger(__name__)


@dataclass
class Article:
    """A validated news article from web search."""
    url: str
    outlet: str = ""
    title: str = ""
    published_date: str = ""
    ingested_at: str = ""
    content: str = ""
    credibility_score: float = 0.0  # Placeholder — filled by CredibilityScorer
    metadata_warning: str = None
    score: float = 0.0


class WebSearchAgent:
    """Tavily-powered web search with validation and dedup."""

    def __init__(self):
        self._client = TavilyClient(api_key=TAVILY_API_KEY)
        self._validator = DataValidator()

    def search(self, query: str, sub_queries: list[str] = None) -> list[Article]:
        """
        Search Tavily for news articles related to query and sub-queries.

        Args:
            query: Main search query.
            sub_queries: Additional sub-queries for broader coverage.

        Returns:
            List of validated, deduplicated Article objects.
        """
        all_raw_results = []

        # Search main query
        queries = [query] + (sub_queries or [])

        for q in queries:
            try:
                response = self._client.search(
                    query=q,
                    max_results=5,
                    search_depth="basic",
                    include_raw_content=False,
                )
                results = response.get("results", [])
                all_raw_results.extend(results)
                logger.info(f"Tavily query '{q[:50]}' returned {len(results)} results")
            except Exception as e:
                logger.error(f"Tavily search failed for '{q[:50]}': {e}")

        if not all_raw_results:
            logger.warning("Tavily returned 0 total results")
            return []

        # Validate batch — per-batch dedup, NOT global
        seen_hashes = set()
        valid, rejected = self._validator.validate_batch(all_raw_results, seen_hashes)

        logger.info(
            f"Web search: {len(valid)} valid, {len(rejected)} rejected "
            f"from {len(all_raw_results)} raw results"
        )

        # Convert to Article dataclasses
        articles = []
        now = datetime.now(timezone.utc).isoformat()

        for item in valid:
            # Extract outlet from URL
            outlet = self._extract_outlet(item.get("url", ""))

            articles.append(Article(
                url=item["url"],
                outlet=outlet,
                title=item.get("title", ""),
                published_date=item.get("published_date", ""),
                ingested_at=now,
                content=item.get("content", ""),
                metadata_warning=item.get("metadata_warning"),
                score=item.get("score", 0.0),
            ))

        return articles

    @staticmethod
    def _extract_outlet(url: str) -> str:
        """Extract outlet name from URL domain."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain
        except Exception:
            return ""

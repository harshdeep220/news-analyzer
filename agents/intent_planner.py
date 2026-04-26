"""
Intent + Query Planner — Combined intent classification and sub-query generation.

Single Flash call → {intent, sub_queries, search_needed, entity_heavy}

Intents:
  - chitchat → respond directly, skip all agents
  - followup → RAG-first (skip Tavily if cache hot)
  - new_research → full pipeline
"""

import json
import logging
from dataclasses import dataclass, field

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL

logger = logging.getLogger(__name__)


@dataclass
class PlanResult:
    """Result of intent classification and query planning."""
    intent: str = "new_research"  # chitchat | followup | new_research
    sub_queries: list[str] = field(default_factory=list)
    search_needed: bool = True
    entity_heavy: bool = False
    direct_response: str = ""  # Only for chitchat


class IntentPlanner:
    """Combined intent classifier and sub-query planner."""

    def __init__(self):
        self._client = GoogleClient()

    def plan(self, query: str, history_summary: str = "") -> PlanResult:
        """
        Classify intent and generate sub-queries in a single Flash call.

        Args:
            query: The user's current query.
            history_summary: Compressed conversation history (max 500 words).

        Returns:
            PlanResult with intent, sub-queries, and flags.
        """
        history_context = ""
        if history_summary:
            history_context = f"\n\nConversation history (summary):\n{history_summary}"

        prompt = f"""Analyze this user query and return a JSON response.

User query: "{query}"{history_context}

Classify the intent and generate search sub-queries.

Return ONLY this JSON (no other text):
{{
    "intent": "chitchat|followup|new_research",
    "sub_queries": ["sub-query-1", "sub-query-2"],
    "search_needed": true/false,
    "entity_heavy": true/false,
    "direct_response": "only if chitchat, otherwise empty string"
}}

Rules:
- "chitchat": greetings, small talk, meta-questions about the system. Set search_needed=false and provide a direct_response.
- "followup": query clearly references or builds on a previous topic in the history. Generate 1-2 sub-queries.
- "new_research": fresh topic requiring web search. Generate 2-3 focused sub-queries.
- "entity_heavy": true if query contains specific names, organizations, places, or events.
- "sub_queries": decompose the main query into specific, searchable sub-queries.

Return ONLY valid JSON."""

        try:
            response = self._client.generate(
                prompt=prompt,
                model=FAST_MODEL,
                system=(
                    "You are a query intent classifier for a news research system. "
                    "You classify queries and generate focused sub-queries for web search."
                ),
            )

            return self._parse_plan(response, query)

        except Exception as e:
            logger.error(f"Intent planning failed: {e}. Defaulting to new_research.")
            return PlanResult(
                intent="new_research",
                sub_queries=[query],
                search_needed=True,
                entity_heavy=False,
            )

    def _parse_plan(self, response: str, original_query: str) -> PlanResult:
        """Parse LLM response into PlanResult with fallback."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            intent = parsed.get("intent", "new_research")
            if intent not in ("chitchat", "followup", "new_research"):
                intent = "new_research"

            result = PlanResult(
                intent=intent,
                sub_queries=parsed.get("sub_queries", [original_query]),
                search_needed=parsed.get("search_needed", True),
                entity_heavy=parsed.get("entity_heavy", False),
                direct_response=parsed.get("direct_response", ""),
            )

            logger.info(
                f"Intent: {result.intent}, sub_queries: {len(result.sub_queries)}, "
                f"entity_heavy: {result.entity_heavy}"
            )
            return result

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse intent plan: {e}")
            return PlanResult(
                intent="new_research",
                sub_queries=[original_query],
                search_needed=True,
                entity_heavy=False,
            )

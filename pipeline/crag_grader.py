"""
CRAG Grader — Corrective RAG document grading.

Grades each retrieved document as relevant/partial/irrelevant in a SINGLE
Flash API call (batch grading). Routes to tavily_only if all docs score low,
or if session has < 5 docs (cold session — zero API calls).
"""

import json
import logging

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL, CRAG_COLD_SESSION_THRESHOLD

logger = logging.getLogger(__name__)


class GradeResult:
    """Result of CRAG grading."""
    def __init__(self, grades: list[dict], routing_decision: str):
        self.grades = grades  # [{doc_id, grade}]
        self.routing_decision = routing_decision  # rag_only | rag_plus_tavily | tavily_only


class CRAGGrader:
    """Batch document relevance grader using single Flash call."""

    def __init__(self):
        self._client = GoogleClient()

    def grade_documents(
        self, query: str, docs: list, collection_count: int = None
    ) -> GradeResult:
        """
        Grade documents for relevance to query.

        Args:
            query: The user's query.
            docs: List of Chunk objects from retrieval.
            collection_count: Total docs in collection (for cold session check).

        Returns:
            GradeResult with per-doc grades and routing decision.
        """
        # Cold session → tavily_only, zero API calls
        if collection_count is not None and collection_count < CRAG_COLD_SESSION_THRESHOLD:
            logger.info(
                f"Cold session ({collection_count} docs < {CRAG_COLD_SESSION_THRESHOLD}) "
                f"→ tavily_only routing"
            )
            return GradeResult(
                grades=[{"doc_id": d.id, "grade": "irrelevant"} for d in docs],
                routing_decision="tavily_only",
            )

        if not docs:
            return GradeResult(grades=[], routing_decision="tavily_only")

        # Build batch prompt — ONE API call for all docs
        doc_entries = []
        for i, doc in enumerate(docs):
            truncated_text = doc.text[:500]  # Keep prompt manageable
            doc_entries.append(f"DOC_{i} (id={doc.id}):\n{truncated_text}")

        docs_text = "\n\n".join(doc_entries)

        prompt = f"""Grade the relevance of each document to the query.
For each document, assign: "relevant", "partial", or "irrelevant".

Query: {query}

Documents:
{docs_text}

Return a JSON array with grades for each document:
[{{"doc_id": "DOC_0_ID", "grade": "relevant|partial|irrelevant"}}]

Return ONLY the JSON array, no other text."""

        try:
            response = self._client.generate(
                prompt=prompt,
                model=FAST_MODEL,
                system="You are a document relevance grader. Grade strictly but fairly.",
            )

            # Parse grades
            grades = self._parse_grades(response, docs)

        except Exception as e:
            logger.error(f"CRAG grading failed: {e}. Defaulting to rag_plus_tavily.")
            grades = [{"doc_id": d.id, "grade": "partial"} for d in docs]

        # Determine routing
        grade_values = [g["grade"] for g in grades]
        relevant_count = grade_values.count("relevant") + grade_values.count("partial")

        if relevant_count == 0:
            routing = "tavily_only"
        elif relevant_count < len(grades):
            routing = "rag_plus_tavily"
        else:
            routing = "rag_only"

        logger.info(
            f"CRAG grading: {len(grades)} docs → "
            f"relevant={grade_values.count('relevant')}, "
            f"partial={grade_values.count('partial')}, "
            f"irrelevant={grade_values.count('irrelevant')} "
            f"→ {routing}"
        )

        return GradeResult(grades=grades, routing_decision=routing)

    def _parse_grades(self, response: str, docs: list) -> list[dict]:
        """Parse LLM response into grade list, with fallback."""
        try:
            # Clean response — extract JSON
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            if isinstance(parsed, list) and len(parsed) == len(docs):
                grades = []
                for i, entry in enumerate(parsed):
                    grade = entry.get("grade", "partial").lower()
                    if grade not in ("relevant", "partial", "irrelevant"):
                        grade = "partial"
                    grades.append({"doc_id": docs[i].id, "grade": grade})
                return grades

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.warning(f"Failed to parse CRAG grades: {e}")

        # Fallback: all partial
        return [{"doc_id": d.id, "grade": "partial"} for d in docs]

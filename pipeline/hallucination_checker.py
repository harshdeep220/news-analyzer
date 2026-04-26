"""
Hallucination Checker — Verifies synthesis claims against source chunks.

Checks each substantive claim in the synthesis against the retrieved source
chunks. Claims not grounded in any source are flagged as ungrounded.

grounding_score tooltip: "Measures whether claims are supported by
retrieved sources, not absolute factual accuracy."
"""

import json
import logging
from dataclasses import dataclass, field

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL

logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    """Result of hallucination checking."""
    grounded_claims: list[str] = field(default_factory=list)
    ungrounded_claims: list[str] = field(default_factory=list)
    grounding_score: float = 1.0  # 0.0 = all hallucinated, 1.0 = all grounded


class HallucinationChecker:
    """Verifies synthesis claims against retrieved source chunks."""

    def __init__(self):
        self._client = GoogleClient()

    def check(self, synthesis, source_chunks: list) -> HallucinationReport:
        """
        Check synthesis for hallucinated claims.

        Args:
            synthesis: SynthesisResult with answer text.
            source_chunks: List of Chunk objects used during synthesis.

        Returns:
            HallucinationReport with grounded/ungrounded claims and score.
        """
        answer = synthesis.answer if hasattr(synthesis, "answer") else str(synthesis)

        if not answer or not source_chunks:
            return HallucinationReport(grounding_score=0.5)

        # Build source context
        source_texts = []
        for i, chunk in enumerate(source_chunks[:10]):  # Cap at 10 chunks
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            source_texts.append(f"SOURCE_{i}:\n{text[:500]}")

        sources_combined = "\n\n".join(source_texts)

        try:
            response = self._client.generate(
                prompt=(
                    f"Check if the claims in this synthesis are supported by the sources.\n\n"
                    f"SYNTHESIS:\n{answer}\n\n"
                    f"SOURCES:\n{sources_combined}\n\n"
                    f"Extract the substantive factual claims from the synthesis. "
                    f"For each claim, check if it is supported by at least one source. "
                    f"Ignore stylistic statements, transitions, and hedging language — "
                    f"only check factual claims (names, numbers, events, dates, quotes).\n\n"
                    f"Return ONLY this JSON:\n"
                    f'{{\n'
                    f'  "grounded_claims": ["claim clearly supported by sources"],\n'
                    f'  "ungrounded_claims": ["claim NOT supported by any source"]\n'
                    f'}}\n\n'
                    f"Return ONLY valid JSON."
                ),
                model=FAST_MODEL,
                system=(
                    "You are a factual grounding checker. Identify claims in a synthesis "
                    "and verify each against provided sources. Be precise about what "
                    "constitutes a substantive factual claim vs. stylistic language."
                ),
            )

            return self._parse_report(response)

        except Exception as e:
            logger.error(f"Hallucination check failed: {e}")
            return HallucinationReport(grounding_score=0.5)

    def _parse_report(self, response: str) -> HallucinationReport:
        """Parse LLM response into HallucinationReport."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            grounded = parsed.get("grounded_claims", [])
            ungrounded = parsed.get("ungrounded_claims", [])

            total = len(grounded) + len(ungrounded)
            score = len(grounded) / total if total > 0 else 1.0

            return HallucinationReport(
                grounded_claims=grounded,
                ungrounded_claims=ungrounded,
                grounding_score=round(score, 2),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse hallucination report: {e}")
            return HallucinationReport(grounding_score=0.5)

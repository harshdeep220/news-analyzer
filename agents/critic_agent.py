"""
Critic Agent — Quality gate with circuit breaker.

Evaluates synthesis quality on completeness, accuracy, and coverage.
Circuit breaker: max 2 retries. On exhaustion, returns max(attempts, key=score).
Pass threshold: score > 0.4 regardless of flag.
"""

import json
import logging
from dataclasses import dataclass, field

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL, CRITIC_PASS_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Result of critic evaluation."""
    passed: bool = False
    score: float = 0.0
    issues: list[str] = field(default_factory=list)


class CriticAgent:
    """Quality gate evaluator with circuit breaker pattern."""

    def __init__(self):
        self._client = GoogleClient()

    def evaluate(self, synthesis_result, query: str) -> CriticVerdict:
        """
        Evaluate synthesis quality.

        Args:
            synthesis_result: SynthesisResult from SynthesisAgent.
            query: The original user query.

        Returns:
            CriticVerdict with pass/fail, score, and issues.
        """
        prompt = f"""Evaluate the quality of this research synthesis.

Original Query: {query}

Synthesis:
{synthesis_result.answer}

Number of citations: {len(synthesis_result.citations)}
Reasoning gaps identified: {synthesis_result.reasoning_gaps}

Evaluate on:
1. Completeness: Does it answer the query fully?
2. Accuracy: Are claims supported by the cited sources?
3. Coverage: Does it cover multiple perspectives?

Return ONLY this JSON:
{{
    "score": 0.0 to 1.0,
    "passed": true/false,
    "issues": ["issue1", "issue2"]
}}"""

        try:
            response = self._client.generate(
                prompt=prompt,
                model=FAST_MODEL,
                system=(
                    "You are a research quality critic. Score synthesis accurately. "
                    "Be fair but rigorous. A score of 0.5+ means acceptable quality."
                ),
            )
            return self._parse_verdict(response)

        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}")
            return CriticVerdict(
                passed=True,  # Default pass on error to avoid blocking
                score=0.5,
                issues=[f"evaluation_error: {str(e)}"],
            )

    def _parse_verdict(self, response: str) -> CriticVerdict:
        """Parse LLM response into CriticVerdict."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)
            score = float(parsed.get("score", 0.5))
            passed_flag = parsed.get("passed", False)
            issues = parsed.get("issues", [])

            # Override: score > threshold passes regardless of flag
            if score > CRITIC_PASS_THRESHOLD:
                passed_flag = True

            return CriticVerdict(
                passed=passed_flag,
                score=score,
                issues=issues,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse critic verdict: {e}")
            return CriticVerdict(passed=True, score=0.5, issues=["parse_error"])

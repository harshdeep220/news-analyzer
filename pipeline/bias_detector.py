"""
Bias Detector — Detects political lean and loaded language in articles.

Returns: lean (7-point scale), confidence, bias_types, loaded_language, missing_perspectives.
Zero API calls if word_count < 250 → returns insufficient_content.
"""

import json
import logging
from dataclasses import dataclass, field

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL, BIAS_MIN_WORDS

logger = logging.getLogger(__name__)


@dataclass
class BiasResult:
    """Bias detection result."""
    lean: str = "insufficient_content"  # Far-Left|Left|Center-Left|Center|Center-Right|Right|Far-Right|insufficient_content
    confidence: float = 0.0
    bias_types: list[str] = field(default_factory=list)
    loaded_language: list[str] = field(default_factory=list)
    missing_perspectives: str = ""
    url: str = ""


VALID_LEANS = [
    "Far-Left", "Left", "Center-Left", "Center",
    "Center-Right", "Right", "Far-Right",
    "insufficient_content",
]


class BiasDetector:
    """Detects political lean and bias indicators in article content."""

    def __init__(self):
        self._client = GoogleClient()

    def detect(self, article: dict) -> BiasResult:
        """
        Detect bias in an article.

        Args:
            article: Dict with content, url, outlet.

        Returns:
            BiasResult with lean, confidence, and bias indicators.
        """
        content = article.get("content", "")
        url = article.get("url", "")
        word_count = len(content.split())

        # Zero API calls if content too short
        if word_count < BIAS_MIN_WORDS:
            logger.debug(
                f"Bias skip: {word_count} words < {BIAS_MIN_WORDS} minimum ({url[:60]})"
            )
            return BiasResult(
                lean="insufficient_content",
                confidence=0.0,
                url=url,
            )

        try:
            response = self._client.generate(
                prompt=(
                    f"Analyze the political lean and bias of this article.\n\n"
                    f"Article ({word_count} words):\n{content[:2000]}\n\n"
                    f"Return ONLY this JSON:\n"
                    f'{{\n'
                    f'  "lean": "Far-Left|Left|Center-Left|Center|Center-Right|Right|Far-Right",\n'
                    f'  "confidence": 0.0 to 1.0,\n'
                    f'  "bias_types": ["selection_bias", "framing_bias", "omission_bias"],\n'
                    f'  "loaded_language": ["word1", "phrase2"],\n'
                    f'  "missing_perspectives": "Brief note on what perspectives are absent"\n'
                    f'}}\n\n'
                    f"Rules:\n"
                    f"- Be fair and measured. This is classification, not accusation.\n"
                    f"- confidence should reflect how clear the lean signal is.\n"
                    f"- loaded_language: max 5 specific words/phrases from the text.\n"
                    f"- Return ONLY valid JSON."
                ),
                model=FAST_MODEL,
                system=(
                    "You are a media bias analyst. You classify articles on a 7-point "
                    "political spectrum with measured confidence. Be fair and evidence-based."
                ),
            )

            return self._parse_result(response, url)

        except Exception as e:
            logger.error(f"Bias detection failed for {url[:60]}: {e}")
            return BiasResult(lean="Center", confidence=0.3, url=url)

    def _parse_result(self, response: str, url: str) -> BiasResult:
        """Parse LLM response into BiasResult."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            lean = parsed.get("lean", "Center")
            if lean not in VALID_LEANS:
                lean = "Center"

            return BiasResult(
                lean=lean,
                confidence=float(parsed.get("confidence", 0.5)),
                bias_types=parsed.get("bias_types", []),
                loaded_language=parsed.get("loaded_language", [])[:5],
                missing_perspectives=parsed.get("missing_perspectives", ""),
                url=url,
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse bias result: {e}")
            return BiasResult(lean="Center", confidence=0.3, url=url)

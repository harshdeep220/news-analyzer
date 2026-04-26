"""
Formatter Agent — Renders synthesis results as validated HTML.

Produces:
  - answer_html: Main answer with inline citations
  - citations_html: Source list with hyperlinks
  - credibility_badges_html: Green/amber/red badges per source
  - bias_panel_html: Lean + confidence per source
  - source_comparison_html: Agreed/disputed facts

All HTML validated through html.parser. Plain-text fallback on parse error.
"""

import logging
from dataclasses import dataclass, field
from html.parser import HTMLParser

from infrastructure.google_client import GoogleClient
from config import FAST_MODEL

logger = logging.getLogger(__name__)


class _HTMLValidator(HTMLParser):
    """Validates HTML by parsing — raises on malformed input."""
    def __init__(self):
        super().__init__()
        self.valid = True
        self.error_msg = ""

    def handle_starttag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        pass

    def error(self, message):
        self.valid = False
        self.error_msg = message


def _validate_html(html_str: str) -> bool:
    """Check if HTML string is valid."""
    try:
        parser = _HTMLValidator()
        parser.feed(html_str)
        return parser.valid
    except Exception:
        return False


def _safe_html(html_str: str, fallback_text: str = "") -> str:
    """Return html_str if valid, otherwise return plain-text fallback."""
    if _validate_html(html_str):
        return html_str
    logger.warning("Malformed HTML detected — using plain-text fallback")
    return f"<p>{fallback_text or html_str}</p>"


@dataclass
class FormattedResponse:
    """Formatted HTML response components."""
    answer_html: str = ""
    citations_html: str = ""
    credibility_badges_html: str = ""
    bias_panel_html: str = ""
    source_comparison_html: str = ""


class FormatterAgent:
    """Renders pipeline results as validated HTML."""

    def __init__(self):
        self._client = GoogleClient()

    def format(
        self,
        synthesis,
        credibility_map: dict = None,
        bias_map: dict = None,
        source_comparison=None,
        hallucination_report=None,
    ) -> FormattedResponse:
        """
        Format synthesis and intelligence results as HTML.

        Args:
            synthesis: SynthesisResult from SynthesisAgent.
            credibility_map: Dict of url → CredibilityScore (Phase 2).
            bias_map: Dict of url → BiasResult (Phase 2).
            source_comparison: ComparisonResult (Phase 2).
            hallucination_report: HallucinationReport (Phase 2).

        Returns:
            FormattedResponse with validated HTML fragments.
        """
        credibility_map = credibility_map or {}
        bias_map = bias_map or {}

        # ─── Main answer ─────────────────────────────────────────────
        answer_html = self._format_answer(synthesis)

        # ─── Citations ───────────────────────────────────────────────
        citations_html = self._format_citations(synthesis.citations)

        # ─── Credibility badges ───────────────────────────────────────
        credibility_badges_html = self._format_credibility_badges(credibility_map)

        # ─── Bias panel ──────────────────────────────────────────────
        bias_panel_html = self._format_bias_panel(bias_map)

        # ─── Source comparison ───────────────────────────────────────
        source_comparison_html = ""
        if source_comparison:
            source_comparison_html = self._format_comparison(source_comparison)

        # ─── Hallucination warnings ──────────────────────────────────
        if hallucination_report and hasattr(hallucination_report, "ungrounded_claims"):
            if hallucination_report.ungrounded_claims:
                warnings = "".join(
                    f'<li class="ungrounded-claim">⚠ {claim}</li>'
                    for claim in hallucination_report.ungrounded_claims
                )
                answer_html += _safe_html(
                    f'<div class="hallucination-warnings">'
                    f'<h4>⚠ Ungrounded Claims</h4><ul>{warnings}</ul>'
                    f'<p class="tooltip">Measures whether claims are supported by '
                    f'retrieved sources, not absolute factual accuracy.</p></div>',
                    "Some claims could not be verified against sources.",
                )

        return FormattedResponse(
            answer_html=answer_html,
            citations_html=citations_html,
            credibility_badges_html=credibility_badges_html,
            bias_panel_html=bias_panel_html,
            source_comparison_html=source_comparison_html,
        )

    def _format_answer(self, synthesis) -> str:
        """Format the main synthesis answer as HTML."""
        answer = synthesis.answer
        # Convert markdown-ish formatting to HTML
        paragraphs = answer.split("\n\n")
        html_parts = []
        for p in paragraphs:
            if p.strip():
                html_parts.append(f"<p>{p.strip()}</p>")
        html = "\n".join(html_parts) if html_parts else f"<p>{answer}</p>"
        return _safe_html(html, answer)

    def _format_citations(self, citations: list[dict]) -> str:
        """Format citations as HTML list with hyperlinks."""
        if not citations:
            return ""

        items = []
        for c in citations:
            url = c.get("source_url", "#")
            text = c.get("chunk_text", "")[:150]
            items.append(
                f'<li><a href="{url}" target="_blank" rel="noopener">{url}</a>'
                f'<p class="citation-excerpt">{text}...</p></li>'
            )

        html = f'<div class="citations"><h4>Sources</h4><ol>{"".join(items)}</ol></div>'
        return _safe_html(html, f"Sources: {len(citations)} cited")

    def _format_credibility_badges(self, credibility_map: dict) -> str:
        """Format credibility badges — green (>70), amber (40-70), red (<40)."""
        if not credibility_map:
            return ""

        badges = []
        for url, score_obj in credibility_map.items():
            score = score_obj if isinstance(score_obj, (int, float)) else getattr(score_obj, "total", 0)
            if score > 70:
                color_class = "cred-green"
            elif score >= 40:
                color_class = "cred-amber"
            else:
                color_class = "cred-red"

            domain = url.split("//")[-1].split("/")[0] if "//" in url else url
            badges.append(
                f'<span class="cred-badge {color_class}" '
                f'title="Credibility score: {score}/100">'
                f'{domain} ({score})</span>'
            )

        html = f'<div class="credibility-badges">{"  ".join(badges)}</div>'
        return _safe_html(html)

    def _format_bias_panel(self, bias_map: dict) -> str:
        """Format bias indicators with lean + confidence %."""
        if not bias_map:
            return ""

        items = []
        for url, bias_obj in bias_map.items():
            lean = bias_obj if isinstance(bias_obj, str) else getattr(bias_obj, "lean", "unknown")
            confidence = getattr(bias_obj, "confidence", 0) if hasattr(bias_obj, "confidence") else 0
            conf_pct = f"{confidence * 100:.0f}%" if isinstance(confidence, float) else str(confidence)

            if lean == "insufficient_content":
                items.append(
                    f'<span class="bias-indicator bias-grey" '
                    f'title="Automated classification. May not reflect full context.">'
                    f'Insufficient content</span>'
                )
            else:
                domain = url.split("//")[-1].split("/")[0] if "//" in url else url
                items.append(
                    f'<span class="bias-indicator" '
                    f'title="Automated classification. May not reflect full context.">'
                    f'Detected lean: {lean} ({conf_pct} confidence)</span>'
                )

        html = f'<div class="bias-panel">{"  ".join(items)}</div>'
        return _safe_html(html)

    def _format_comparison(self, comparison) -> str:
        """Format source comparison panel."""
        parts = ['<div class="source-comparison"><h4>Source Comparison</h4>']

        if hasattr(comparison, "agreed_facts") and comparison.agreed_facts:
            facts = "".join(f"<li>{f}</li>" for f in comparison.agreed_facts)
            parts.append(f"<h5>Agreed Facts</h5><ul>{facts}</ul>")

        if hasattr(comparison, "disputed_facts") and comparison.disputed_facts:
            facts = "".join(f"<li>{f}</li>" for f in comparison.disputed_facts)
            parts.append(f"<h5>Disputed Facts</h5><ul>{facts}</ul>")

        if hasattr(comparison, "framing_differences") and comparison.framing_differences:
            parts.append(f"<p><strong>Framing:</strong> {comparison.framing_differences}</p>")

        parts.append("</div>")
        html = "\n".join(parts)
        return _safe_html(html, "Source comparison available")

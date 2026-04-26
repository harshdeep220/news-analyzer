"""
Synthesis Agent — Deep reasoning synthesis using gemini-2.5-pro.

Iterative RAG: up to 2 passes if gaps detected. Gap must be > 5 words
to justify a new pass. Thinking tokens handled natively — no stripping.
"""

import json
import logging
from dataclasses import dataclass, field

from infrastructure.google_client import GoogleClient
from config import SYNTH_MODEL, SYNTHESIS_MAX_PASSES, SYNTHESIS_GAP_MIN_WORDS

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of synthesis with citations and gap analysis."""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)  # [{source_url, chunk_text}]
    critic_score: float = 0.0  # Placeholder — filled by CriticAgent
    reasoning_gaps: list[str] = field(default_factory=list)
    pass_count: int = 1


class SynthesisAgent:
    """Gemini 2.5 Pro-powered synthesis with iterative retrieval."""

    def __init__(self):
        self._client = GoogleClient()

    def synthesize(
        self,
        context_chunks: list,
        query: str,
        history: str = "",
        retriever=None,
        collection=None,
        session_id: str = None,
    ) -> SynthesisResult:
        """
        Synthesize an answer from retrieved context chunks.

        Args:
            context_chunks: List of Chunk objects from retrieval.
            query: The user's query.
            history: Compressed conversation history.
            retriever: HybridRetriever for iterative passes (optional).
            collection: ChromaDB collection for iterative passes (optional).
            session_id: Session ID for iterative passes (optional).

        Returns:
            SynthesisResult with answer, citations, and gap analysis.
        """
        current_chunks = list(context_chunks)
        result = None

        for pass_num in range(1, SYNTHESIS_MAX_PASSES + 1):
            result = self._single_pass(current_chunks, query, history, pass_num)

            # Check if iterative pass is needed
            gaps = result.reasoning_gaps
            substantial_gaps = [
                g for g in gaps if len(g.split()) > SYNTHESIS_GAP_MIN_WORDS
            ]

            if not substantial_gaps or pass_num >= SYNTHESIS_MAX_PASSES:
                break

            # Attempt iterative retrieval for gaps
            if retriever and collection and session_id:
                logger.info(
                    f"Synthesis pass {pass_num}: {len(substantial_gaps)} gaps → "
                    f"retrieving more context"
                )
                for gap in substantial_gaps[:2]:  # Max 2 gap queries
                    try:
                        additional = retriever.retrieve(
                            query=gap,
                            collection=collection,
                            session_id=session_id,
                            top_k=3,
                        )
                        # Add to context, avoiding duplicates
                        existing_ids = {c.id for c in current_chunks}
                        for chunk in additional:
                            if chunk.id not in existing_ids:
                                current_chunks.append(chunk)
                                existing_ids.add(chunk.id)
                    except Exception as e:
                        logger.warning(f"Gap retrieval failed for '{gap[:50]}': {e}")
            else:
                break

        result.pass_count = pass_num
        return result

    def _single_pass(
        self, chunks: list, query: str, history: str, pass_num: int
    ) -> SynthesisResult:
        """Execute a single synthesis pass."""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk.metadata.get("source", chunk.metadata.get("url", f"source_{i}"))
            date = chunk.metadata.get("published_date", "unknown date")
            outlet = chunk.metadata.get("outlet", "unknown outlet")
            header = f"[Source: {outlet} | Date: {date} | URL: {source}]"
            context_parts.append(f"{header}\n{chunk.text}")

        context_text = "\n\n---\n\n".join(context_parts)

        history_section = ""
        if history:
            history_section = f"\n\nConversation context:\n{history}"

        prompt = f"""Based on the following source documents, provide a comprehensive answer to the query.

Query: {query}{history_section}

Source Documents:
{context_text}

Instructions:
1. Synthesize information from multiple sources into a coherent answer.
2. Cite specific sources using [Source: outlet_name] markers.
3. If you find gaps in the available information, list them.
4. Be factual and precise. Do not add information not present in the sources.

Return your response as JSON:
{{
    "answer": "Your comprehensive synthesized answer with [Source: outlet] citations",
    "citations": [
        {{"source_url": "url", "chunk_text": "relevant excerpt"}}
    ],
    "reasoning_gaps": ["gap1 if any", "gap2 if any"]
}}

Return ONLY valid JSON."""

        try:
            response = self._client.generate(
                prompt=prompt,
                model=SYNTH_MODEL,
                system=(
                    "You are a news research synthesizer. You combine multiple sources "
                    "into comprehensive, well-cited answers. Be thorough but concise."
                ),
            )
            return self._parse_synthesis(response, chunks)

        except Exception as e:
            logger.error(f"Synthesis pass {pass_num} failed: {e}")
            # Fallback: concatenate chunk summaries
            fallback_answer = f"Based on {len(chunks)} sources: "
            fallback_answer += " | ".join(
                c.text[:200] for c in chunks[:3]
            )
            return SynthesisResult(
                answer=fallback_answer,
                citations=[],
                reasoning_gaps=["synthesis_failed"],
            )

    def _parse_synthesis(self, response: str, chunks: list) -> SynthesisResult:
        """Parse LLM response into SynthesisResult."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)

            return SynthesisResult(
                answer=parsed.get("answer", response),
                citations=parsed.get("citations", []),
                reasoning_gaps=parsed.get("reasoning_gaps", []),
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse synthesis JSON: {e}")
            # Use raw response as answer
            return SynthesisResult(
                answer=response.strip(),
                citations=[],
                reasoning_gaps=[],
            )

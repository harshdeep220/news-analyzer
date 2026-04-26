"""
Reranker — Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.

Takes top-k candidates from HybridRetriever and reranks to top-n
using a cross-encoder model running on CPU.

Default: top_k=10 input → top_n=5 output.
"""

import logging
import time

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RERANKER_TOP_N

logger = logging.getLogger(__name__)

# ─── Module-level model (lazy loaded) ────────────────────────────────────────
_model = None


def _get_model():
    """Lazy-load the cross-encoder model."""
    global _model
    if _model is None:
        start = time.time()
        _model = CrossEncoder(RERANKER_MODEL)
        logger.info(f"Reranker model loaded in {time.time() - start:.2f}s")
    return _model


def rerank(query: str, chunks: list, top_n: int = None) -> list:
    """
    Rerank chunks using cross-encoder model.

    Args:
        query: The search query.
        chunks: List of Chunk objects from HybridRetriever.
        top_n: Number of top results to return (default: RERANKER_TOP_N).

    Returns:
        Reranked list of Chunk objects, sorted by rerank_score descending.
    """
    top_n = top_n or RERANKER_TOP_N

    if not chunks:
        return []

    if len(chunks) <= top_n:
        return chunks

    model = _get_model()

    # Build query-doc pairs
    pairs = [(query, chunk.text) for chunk in chunks]

    start = time.time()
    scores = model.predict(pairs)
    elapsed = time.time() - start

    # Assign scores and sort
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = float(score)

    reranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]

    logger.info(
        f"Reranked {len(chunks)} → {len(reranked)} chunks in {elapsed:.2f}s "
        f"(best: {reranked[0].rerank_score:.4f}, worst: {reranked[-1].rerank_score:.4f})"
    )

    return reranked

"""
Hybrid Retriever — BM25 + Semantic + Reciprocal Rank Fusion (RRF).

Combines BM25 keyword search with ChromaDB semantic search using RRF
for optimal retrieval quality on news content (named entities + meaning).

BM25 index cached per session_id. Cache invalidated when collection count changes.
"""

import logging
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from infrastructure.google_client import GoogleClient
from config import RERANKER_TOP_K

logger = logging.getLogger(__name__)

# ─── Module-level BM25 cache ─────────────────────────────────────────────────
# Key = (session_id, doc_count), Value = (BM25Okapi instance, doc_ids, doc_texts)
_bm25_cache: dict[tuple[str, int], tuple] = {}


@dataclass
class Chunk:
    """A retrieved document chunk with metadata."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class HybridRetriever:
    """BM25 + Semantic fusion retriever with RRF scoring."""

    def __init__(self):
        self._google_client = GoogleClient()

    def retrieve(
        self,
        query: str,
        collection,
        session_id: str,
        top_k: int = None,
        entity_heavy: bool = False,
    ) -> list[Chunk]:
        """
        Retrieve top-k chunks using hybrid BM25 + semantic search with RRF.

        Args:
            query: The search query.
            collection: ChromaDB collection for this session.
            session_id: Session ID for BM25 cache keying.
            top_k: Number of candidates (default: RERANKER_TOP_K from config).
            entity_heavy: If True, boost BM25 weight (more keyword-focused).

        Returns:
            List of Chunk objects sorted by RRF score, length = top_k.
        """
        top_k = top_k or RERANKER_TOP_K

        doc_count = collection.count()
        if doc_count == 0:
            logger.info("Collection is empty — returning no results")
            return []

        # Adjust top_k if collection has fewer docs
        effective_k = min(top_k, doc_count)

        # ─── Semantic search ──────────────────────────────────────────
        start = time.time()
        query_embedding = self._google_client.embed(query)
        semantic_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas"],
        )
        semantic_time = time.time() - start

        semantic_chunks = []
        if semantic_results and semantic_results["ids"] and semantic_results["ids"][0]:
            for i, doc_id in enumerate(semantic_results["ids"][0]):
                semantic_chunks.append(Chunk(
                    id=doc_id,
                    text=semantic_results["documents"][0][i],
                    metadata=semantic_results["metadatas"][0][i] if semantic_results["metadatas"] else {},
                ))

        # ─── BM25 search ─────────────────────────────────────────────
        start = time.time()
        bm25, all_ids, all_texts = self._get_bm25_index(collection, session_id, doc_count)

        tokenized_query = query.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        # Get top-k by BM25 score
        scored_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:effective_k]

        bm25_chunks = []
        # Fetch metadata for BM25 results
        if scored_indices:
            bm25_ids = [all_ids[i] for i in scored_indices]
            bm25_data = collection.get(ids=bm25_ids, include=["documents", "metadatas"])
            for i, doc_id in enumerate(bm25_data["ids"]):
                bm25_chunks.append(Chunk(
                    id=doc_id,
                    text=bm25_data["documents"][i],
                    metadata=bm25_data["metadatas"][i] if bm25_data["metadatas"] else {},
                ))
        bm25_time = time.time() - start

        # ─── RRF Fusion ──────────────────────────────────────────────
        # RRF weight: entity_heavy boosts BM25
        bm25_weight = 0.6 if entity_heavy else 0.4
        semantic_weight = 1.0 - bm25_weight

        rrf_k = 60  # Standard RRF constant
        rrf_scores: dict[str, float] = {}
        rrf_chunks: dict[str, Chunk] = {}

        # Score semantic results
        for rank, chunk in enumerate(semantic_chunks):
            score = semantic_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + score
            rrf_chunks[chunk.id] = chunk

        # Score BM25 results
        for rank, chunk in enumerate(bm25_chunks):
            score = bm25_weight * (1.0 / (rrf_k + rank + 1))
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0) + score
            if chunk.id not in rrf_chunks:
                rrf_chunks[chunk.id] = chunk

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for doc_id in sorted_ids[:effective_k]:
            chunk = rrf_chunks[doc_id]
            chunk.rrf_score = rrf_scores[doc_id]
            results.append(chunk)

        logger.info(
            f"Hybrid retrieval: {len(results)} chunks "
            f"(semantic: {semantic_time:.2f}s, BM25: {bm25_time:.2f}s, "
            f"entity_heavy={entity_heavy})"
        )

        return results

    def _get_bm25_index(
        self, collection, session_id: str, doc_count: int
    ) -> tuple:
        """
        Get or build BM25 index for a session. Cached by (session_id, doc_count).

        Args:
            collection: ChromaDB collection.
            session_id: Session ID.
            doc_count: Current document count (cache invalidation key).

        Returns:
            Tuple of (BM25Okapi, doc_ids, doc_texts).
        """
        cache_key = (session_id, doc_count)

        if cache_key in _bm25_cache:
            return _bm25_cache[cache_key]

        # Build new index
        start = time.time()
        all_data = collection.get(include=["documents"])
        all_ids = all_data["ids"]
        all_texts = all_data["documents"]

        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in all_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        elapsed = time.time() - start
        if elapsed > 0.5:
            logger.warning(f"BM25 index rebuild took {elapsed:.2f}s for {doc_count} docs")
        else:
            logger.info(f"BM25 index built in {elapsed:.2f}s for {doc_count} docs")

        # Invalidate old entries for this session
        old_keys = [k for k in _bm25_cache if k[0] == session_id]
        for k in old_keys:
            del _bm25_cache[k]

        _bm25_cache[cache_key] = (bm25, all_ids, all_texts)
        return bm25, all_ids, all_texts

    @staticmethod
    def invalidate_cache(session_id: str) -> None:
        """Invalidate BM25 cache for a session (call when new chunks added)."""
        old_keys = [k for k in _bm25_cache if k[0] == session_id]
        for k in old_keys:
            del _bm25_cache[k]

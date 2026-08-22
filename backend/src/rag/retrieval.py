from typing import Protocol

import numpy as np

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.models import Chunk, ScoredChunk
from backend.src.rag.stores import BM25Index, VectorIndex


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]: ...


class Retriever:
    """Expose vector, hybrid and reranked retrieval through one public API."""

    def __init__(
        self,
        config: RAGConfig,
        embedder: Embedder,
        reranker: Reranker | None = None,
        vector_index: VectorIndex | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        if config.strategy is RetrievalStrategy.RERANKED and reranker is None:
            raise ValueError("Reranked retrieval requires a reranker.")
        self._config = config
        self._embedder = embedder
        self._reranker = reranker
        self._vector = vector_index or VectorIndex()
        self._bm25 = bm25_index or BM25Index()

    def index(self, chunks: list[Chunk]) -> None:
        embeddings = self._embedder.embed_documents([chunk.content for chunk in chunks])
        self._vector.build(chunks, embeddings)
        self._bm25.build(chunks)

    def retrieve(self, query: str) -> list[ScoredChunk]:
        question = query.strip()
        if not question:
            raise ValueError("query must not be empty.")

        if self._config.strategy is RetrievalStrategy.VECTOR:
            return self._vector_results(question, self._config.top_k)

        candidates = self._hybrid_results(
            question,
            self._config.candidate_top_k,
        )
        if self._config.strategy is RetrievalStrategy.HYBRID:
            return _with_ranks(candidates[: self._config.top_k])

        assert self._reranker is not None
        return self._reranker.rerank(
            question,
            candidates,
            self._config.top_k,
        )

    def _vector_results(self, query: str, top_k: int) -> list[ScoredChunk]:
        return self._vector.search(self._embedder.embed_query(query), top_k)

    def _hybrid_results(self, query: str, top_k: int) -> list[ScoredChunk]:
        vector = self._vector_results(query, top_k)
        lexical = self._bm25.search(query, top_k)
        vector_scores = _normalized_score_lookup(vector)
        lexical_scores = _normalized_score_lookup(lexical)
        chunks = {item.chunk.chunk_id: item.chunk for item in [*vector, *lexical]}

        combined = [
            ScoredChunk(
                chunk=chunk,
                score=(
                    self._config.vector_weight * vector_scores.get(chunk_id, 0.0)
                    + self._config.lexical_weight * lexical_scores.get(chunk_id, 0.0)
                ),
            )
            for chunk_id, chunk in chunks.items()
        ]
        return sorted(combined, key=lambda item: item.score, reverse=True)[:top_k]


def _normalized_score_lookup(items: list[ScoredChunk]) -> dict[str, float]:
    if not items:
        return {}
    scores = [item.score for item in items]
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return {item.chunk.chunk_id: 1.0 for item in items}
    return {
        item.chunk.chunk_id: (item.score - minimum) / (maximum - minimum)
        for item in items
    }


def _with_ranks(items: list[ScoredChunk]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=item.chunk, score=item.score, rank=rank)
        for rank, item in enumerate(items, start=1)
    ]

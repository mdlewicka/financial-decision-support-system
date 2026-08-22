"""No-cost deterministic retrieval for the local reference corpus."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from backend.src.rag.chunking import split_documents
from backend.src.rag.loaders import load_reference_corpus
from backend.src.rag.models import Chunk, Document, ScoredChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)


class DeterministicReferenceRetriever:
    """Rank local chunks by transparent lexical overlap without ML models."""

    def __init__(
        self,
        documents: list[Document],
        *,
        top_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._top_k = top_k
        self._chunks = split_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._chunk_tokens = {
            chunk.chunk_id: Counter(_tokens(chunk.content)) for chunk in self._chunks
        }

    @classmethod
    def from_corpus_path(
        cls,
        path: Path,
        *,
        top_k: int = 5,
    ) -> DeterministicReferenceRetriever:
        return cls(
            load_reference_corpus(path),
            top_k=top_k,
        )

    def retrieve(self, question: str) -> list[ScoredChunk]:
        query_tokens = Counter(_tokens(question))
        if not query_tokens:
            raise ValueError("question must contain searchable text.")

        scored = [(self._score(query_tokens, chunk), chunk) for chunk in self._chunks]
        ranked = sorted(
            (item for item in scored if item[0] > 0.0),
            key=lambda item: (-item[0], item[1].chunk_id),
        )[: self._top_k]
        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                rank=rank,
            )
            for rank, (score, chunk) in enumerate(ranked, start=1)
        ]

    def _score(
        self,
        query_tokens: Counter[str],
        chunk: Chunk,
    ) -> float:
        chunk_tokens = self._chunk_tokens[chunk.chunk_id]
        overlap = sum(
            min(count, chunk_tokens.get(token, 0))
            for token, count in query_tokens.items()
        )
        return overlap / sum(query_tokens.values())


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(value)]

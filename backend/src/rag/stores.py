import re

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from backend.src.rag.models import Chunk, ScoredChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class VectorIndex:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: faiss.Index | None = None

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise ValueError("embeddings must have one row per chunk.")
        faiss.normalize_L2(matrix)
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._chunks = list(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[ScoredChunk]:
        if self._index is None:
            raise RuntimeError("Vector index has not been built.")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        return [
            ScoredChunk(
                chunk=self._chunks[int(index)],
                score=float(score),
                rank=rank,
            )
            for rank, (score, index) in enumerate(
                zip(scores[0], indices[0], strict=True),
                start=1,
            )
            if index >= 0
        ]


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._index: BM25Okapi | None = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks must not be empty.")
        self._chunks = list(chunks)
        self._index = BM25Okapi([tokenize(chunk.content) for chunk in chunks])

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if self._index is None:
            raise RuntimeError("BM25 index has not been built.")
        scores = self._index.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            ScoredChunk(
                chunk=self._chunks[int(index)],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(order, start=1)
        ]

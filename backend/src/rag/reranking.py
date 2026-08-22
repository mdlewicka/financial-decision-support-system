from typing import Any

from backend.src.rag.models import ScoredChunk


class CrossEncoderReranker:
    """Rerank retrieved candidates with a lazily loaded cross-encoder."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model: Any = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, item.chunk.content) for item in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        return [
            ScoredChunk(chunk=item.chunk, score=float(score), rank=rank)
            for rank, (item, score) in enumerate(ranked, start=1)
        ]

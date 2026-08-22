from dataclasses import dataclass
from enum import StrEnum


class RetrievalStrategy(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    RERANKED = "reranked"


@dataclass(frozen=True)
class RAGConfig:
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    candidate_top_k: int = 20
    vector_weight: float = 0.5
    lexical_weight: float = 0.5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    language_model: str = "gemma3:4b"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size).")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self.candidate_top_k < self.top_k:
            raise ValueError("candidate_top_k must be at least top_k.")
        if self.vector_weight < 0 or self.lexical_weight < 0:
            raise ValueError("retrieval weights must not be negative.")
        if self.vector_weight + self.lexical_weight == 0:
            raise ValueError("at least one retrieval weight must be positive.")
        for name in ("embedding_model", "reranker_model", "language_model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty.")

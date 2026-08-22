from dataclasses import dataclass, field
from typing import Any


def _required(value: str, label: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{label} must not be empty.")
    return cleaned


@dataclass(frozen=True)
class Document:
    document_id: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_id",
            _required(self.document_id, "document_id"),
        )
        object.__setattr__(self, "content", _required(self.content, "content"))


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _required(self.chunk_id, "chunk_id"))
        object.__setattr__(
            self,
            "document_id",
            _required(self.document_id, "document_id"),
        )
        object.__setattr__(self, "content", _required(self.content, "content"))


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    rank: int = 0

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must not be negative.")


@dataclass(frozen=True)
class Citation:
    document_id: str
    source: str
    title: str | None = None
    page: int | None = None


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    citations: tuple[Citation, ...]
    retrieved_chunks: tuple[ScoredChunk, ...]

from typing import Protocol

from backend.src.rag.chunking import split_documents
from backend.src.rag.config import RAGConfig
from backend.src.rag.models import Citation, Document, RAGAnswer, ScoredChunk
from backend.src.rag.retrieval import Retriever


class Generator(Protocol):
    def generate(self, question: str, context: str) -> str: ...


class RAGPipeline:
    """Index documents, retrieve evidence and return a cited answer."""

    def __init__(
        self,
        config: RAGConfig,
        retriever: Retriever,
        generator: Generator,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._generator = generator

    def index(self, documents: list[Document]) -> int:
        if not documents:
            raise ValueError("documents must not be empty.")
        chunks = split_documents(
            documents,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._retriever.index(chunks)
        return len(chunks)

    def retrieve(self, question: str) -> list[ScoredChunk]:
        return self._retriever.retrieve(question)

    def ask(self, question: str) -> RAGAnswer:
        results = self.retrieve(question)
        if not results:
            raise RuntimeError("No context was retrieved for the question.")
        context = "\n\n".join(
            f"[{item.chunk.document_id}] {item.chunk.content}" for item in results
        )
        answer = self._generator.generate(question, context)
        return RAGAnswer(
            question=question.strip(),
            answer=answer,
            citations=_citations(results),
            retrieved_chunks=tuple(results),
        )


def _citations(results: list[ScoredChunk]) -> tuple[Citation, ...]:
    citations: list[Citation] = []
    seen: set[str] = set()
    for item in results:
        chunk = item.chunk
        if chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        page = chunk.metadata.get("page")
        citations.append(
            Citation(
                document_id=chunk.document_id,
                source=chunk.source,
                title=chunk.metadata.get("title"),
                page=int(page) if page is not None else None,
            )
        )
    return tuple(citations)

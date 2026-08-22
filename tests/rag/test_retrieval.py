import numpy as np
import pytest

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.models import Chunk, ScoredChunk
from backend.src.rag.retrieval import Retriever
from backend.src.rag.stores import BM25Index, VectorIndex, tokenize


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray([self.embed_query(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        text = query.casefold()
        return np.asarray(
            [float("inflation" in text), float("portfolio" in text), 0.1],
            dtype=np.float32,
        )


class ReverseReranker:
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        del query
        return [
            ScoredChunk(item.chunk, item.score, rank)
            for rank, item in enumerate(reversed(candidates[-top_k:]), start=1)
        ]


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk("inflation_0", "inflation", "Inflation and interest rates"),
        Chunk("portfolio_0", "portfolio", "Portfolio diversification"),
        Chunk("risk_0", "risk", "Credit and liquidity risk"),
    ]


def test_tokenize_is_case_insensitive() -> None:
    assert tokenize("Interest RATES") == ["interest", "rates"]


def test_vector_index_returns_ranked_chunks(chunks) -> None:
    embedder = FakeEmbedder()
    index = VectorIndex()
    index.build(chunks, embedder.embed_documents([item.content for item in chunks]))
    results = index.search(embedder.embed_query("inflation"), top_k=2)
    assert results[0].chunk.document_id == "inflation"
    assert [result.rank for result in results] == [1, 2]


def test_vector_index_requires_build() -> None:
    with pytest.raises(RuntimeError, match="not been built"):
        VectorIndex().search(np.ones(3, dtype=np.float32), 1)


def test_bm25_returns_lexical_match(chunks) -> None:
    index = BM25Index()
    index.build(chunks)
    assert index.search("liquidity risk", 1)[0].chunk.document_id == "risk"


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (RetrievalStrategy.VECTOR, "inflation"),
        (RetrievalStrategy.HYBRID, "inflation"),
    ],
)
def test_retriever_supports_vector_and_hybrid(strategy, expected, chunks) -> None:
    retriever = Retriever(
        RAGConfig(strategy=strategy, top_k=1, candidate_top_k=3),
        FakeEmbedder(),
    )
    retriever.index(chunks)
    assert retriever.retrieve("inflation")[0].chunk.document_id == expected


def test_reranked_strategy_uses_reranker(chunks) -> None:
    retriever = Retriever(
        RAGConfig(
            strategy=RetrievalStrategy.RERANKED,
            top_k=1,
            candidate_top_k=3,
        ),
        FakeEmbedder(),
        ReverseReranker(),
    )
    retriever.index(chunks)
    results = retriever.retrieve("inflation")
    assert len(results) == 1
    assert results[0].rank == 1


def test_reranked_strategy_requires_reranker() -> None:
    with pytest.raises(ValueError, match="requires a reranker"):
        Retriever(
            RAGConfig(strategy=RetrievalStrategy.RERANKED),
            FakeEmbedder(),
        )


def test_retriever_rejects_empty_query(chunks) -> None:
    retriever = Retriever(RAGConfig(), FakeEmbedder())
    retriever.index(chunks)
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("  ")

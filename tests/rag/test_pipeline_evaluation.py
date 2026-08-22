import json
from typing import cast

import numpy as np
import pytest

from backend.src.rag.config import RAGConfig, RetrievalStrategy
from backend.src.rag.evaluation import (
    BenchmarkRunner,
    GoldenItem,
    LLMGenerationEvaluator,
    quality_score,
)
from backend.src.rag.factory import build_rag_pipeline
from backend.src.rag.models import (
    Chunk,
    Document,
    RAGAnswer,
    ScoredChunk,
)
from backend.src.rag.pipeline import RAGPipeline
from backend.src.rag.retrieval import Retriever


class FakeRetriever:
    """Deterministic retriever test double."""

    def __init__(
        self,
        results: list[ScoredChunk],
    ) -> None:
        self.results = results
        self.indexed: list[Chunk] = []

    def index(
        self,
        chunks: list[Chunk],
    ) -> None:
        self.indexed = chunks

    def retrieve(
        self,
        query: str,
    ) -> list[ScoredChunk]:
        del query
        return self.results


class FakeGenerator:
    """Deterministic text-generation test double."""

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        return f"Answer to {question} using {context}"


class FakeCompletionModel:
    """Return deterministic generation-evaluation scores."""

    def complete(
        self,
        prompt: str,
    ) -> str:
        del prompt

        return json.dumps(
            {
                "faithfulness": 0.9,
                "context_recall": 0.8,
            }
        )


class SemanticFakeEmbedder:
    """Create deterministic semantic vectors for integration tests."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.asarray(
            [self.embed_query(text) for text in texts],
            dtype=np.float32,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        text = query.casefold()

        return np.asarray(
            [
                float("inflation" in text),
                float("portfolio" in text),
                0.1,
            ],
            dtype=np.float32,
        )


def _as_retriever(
    fake: FakeRetriever,
) -> Retriever:
    """Treat a contract-compatible test double as a Retriever."""
    return cast(
        Retriever,
        fake,
    )


def _result(
    document_id: str,
    rank: int,
) -> ScoredChunk:
    return ScoredChunk(
        Chunk(
            f"{document_id}_chunk_0000",
            document_id,
            f"Evidence from {document_id}",
            source=f"{document_id}.pdf",
            metadata={
                "title": document_id.title(),
            },
        ),
        score=1 / rank,
        rank=rank,
    )


def test_pipeline_indexes_and_returns_citations() -> None:
    retriever = FakeRetriever(
        [
            _result(
                "inflation",
                1,
            ),
            _result(
                "inflation",
                2,
            ),
        ]
    )

    pipeline = RAGPipeline(
        RAGConfig(),
        _as_retriever(retriever),
        FakeGenerator(),
    )

    count = pipeline.index(
        [
            Document(
                "inflation",
                "Inflation evidence",
            )
        ]
    )

    answer = pipeline.ask("What is inflation?")

    assert count == 1

    assert len(answer.citations) == 1

    assert answer.citations[0].source == "inflation.pdf"


def test_pipeline_rejects_empty_document_collection() -> None:
    pipeline = RAGPipeline(
        RAGConfig(),
        _as_retriever(FakeRetriever([])),
        FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="documents",
    ):
        pipeline.index([])


def test_pipeline_requires_retrieved_context() -> None:
    pipeline = RAGPipeline(
        RAGConfig(),
        _as_retriever(FakeRetriever([])),
        FakeGenerator(),
    )

    with pytest.raises(
        RuntimeError,
        match="No context",
    ):
        pipeline.ask("Question?")


def test_compact_rag_runs_end_to_end_with_injected_models() -> None:
    pipeline = build_rag_pipeline(
        RAGConfig(
            strategy=(RetrievalStrategy.HYBRID),
            top_k=1,
            candidate_top_k=2,
        ),
        embedder=(SemanticFakeEmbedder()),
        generator=(FakeGenerator()),
    )

    pipeline.index(
        [
            Document(
                "inflation",
                ("Inflation raises the general price level."),
            ),
            Document(
                "portfolio",
                ("Portfolio diversification reduces risk."),
            ),
        ]
    )

    answer = pipeline.ask("What is inflation?")

    assert answer.citations[0].document_id == "inflation"

    assert "Inflation raises" in answer.answer


def test_llm_generation_evaluator_parses_scores() -> None:
    evaluator = LLMGenerationEvaluator(FakeCompletionModel())

    answer = RAGAnswer(
        "Question?",
        "Answer",
        (),
        (
            _result(
                "doc",
                1,
            ),
        ),
    )

    metrics = evaluator.evaluate(
        GoldenItem(
            "item",
            "Question?",
            ("doc",),
            reference_answer=("Reference"),
        ),
        answer,
    )

    assert metrics.faithfulness == pytest.approx(0.9)

    assert metrics.context_recall == pytest.approx(0.8)


def test_benchmark_calculates_retrieval_and_generation_metrics() -> None:
    pipeline = RAGPipeline(
        RAGConfig(),
        _as_retriever(
            FakeRetriever(
                [
                    _result(
                        "noise",
                        1,
                    ),
                    _result(
                        "expected",
                        2,
                    ),
                ]
            )
        ),
        FakeGenerator(),
    )

    runner = BenchmarkRunner(
        pipeline,
        LLMGenerationEvaluator(FakeCompletionModel()),
    )

    result = runner.run(
        "hybrid",
        [
            GoldenItem(
                "item",
                "Question?",
                ("expected",),
            )
        ],
    )

    item = result.items[0]

    assert item.recall_at_k == 1.0

    assert item.precision_at_k == 0.5

    assert item.reciprocal_rank == 0.5

    assert result.summary.mean_faithfulness == pytest.approx(0.9)

    assert quality_score(result) > 0


def test_benchmark_rejects_empty_dataset() -> None:
    pipeline = RAGPipeline(
        RAGConfig(),
        _as_retriever(FakeRetriever([])),
        FakeGenerator(),
    )

    with pytest.raises(
        ValueError,
        match="items",
    ):
        BenchmarkRunner(pipeline).run(
            "vector",
            [],
        )

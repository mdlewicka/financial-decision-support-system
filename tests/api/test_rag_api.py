import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.src.api.rag_repository import (
    RAGArtifactRepository,
)
from backend.src.api.rag_service import (
    RAGQueryExecutionResult,
    RAGQueryRuntimeError,
)
from backend.src.api.routes.rag import (
    get_rag_query_service,
    get_rag_repository,
)
from backend.src.main import create_app
from backend.src.rag.config import (
    RetrievalStrategy,
)
from backend.src.rag.models import (
    Chunk,
    Citation,
    RAGAnswer,
    ScoredChunk,
)


def _write_corpus(
    path: Path,
) -> None:
    payload = {
        "documents": [
            {
                "document_id": "inflation_basics",
                "content": (
                    "Inflation describes a sustained increase "
                    "in the general price level."
                ),
                "source": ("reference_corpus/inflation_basics"),
                "metadata": {
                    "domain": "macroeconomics",
                    "title": "Inflation Basics",
                },
            },
            {
                "document_id": "financial_risk",
                "content": (
                    "Financial risk includes market, credit and liquidity risk."
                ),
                "source": ("reference_corpus/financial_risk"),
                "metadata": {
                    "domain": "risk",
                    "title": "Financial Risk",
                },
            },
        ]
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_golden_dataset(
    path: Path,
) -> None:
    payload = {
        "items": [
            {
                "item_id": "inflation_question",
                "question": "What is inflation?",
                "expected_document_ids": [
                    "inflation_basics",
                ],
            },
            {
                "item_id": "risk_question",
                "question": "What is financial risk?",
                "expected_document_ids": [
                    "financial_risk",
                ],
            },
        ]
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _ranking_entry(
    *,
    rank: int,
    configuration_name: str,
    quality_score: float,
    duration_seconds: float,
) -> dict[str, object]:
    return {
        "rank": rank,
        "configuration_name": (configuration_name),
        "quality_score": quality_score,
        "item_count": 2,
        "mean_recall_at_k": 1.0,
        "mean_precision_at_k": 0.3,
        "mean_hit_rate": 1.0,
        "mean_reciprocal_rank": 1.0,
        "mean_faithfulness": None,
        "mean_context_recall": None,
        "duration_seconds": (duration_seconds),
    }


def _write_comparison(
    path: Path,
) -> None:
    payload = {
        "best_configuration": "vector",
        "ranking": [
            _ranking_entry(
                rank=1,
                configuration_name="vector",
                quality_score=0.8,
                duration_seconds=2.5,
            ),
            _ranking_entry(
                rank=2,
                configuration_name="hybrid",
                quality_score=0.8,
                duration_seconds=2.0,
            ),
            _ranking_entry(
                rank=3,
                configuration_name="reranked",
                quality_score=0.75,
                duration_seconds=3.0,
            ),
        ],
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _repository(
    tmp_path: Path,
) -> RAGArtifactRepository:
    corpus_path = tmp_path / "reference_corpus.json"
    golden_path = tmp_path / "golden_dataset_expanded.json"
    comparison_path = tmp_path / "comparison.json"

    _write_corpus(corpus_path)
    _write_golden_dataset(golden_path)
    _write_comparison(comparison_path)

    return RAGArtifactRepository(
        corpus_path=corpus_path,
        golden_dataset_path=golden_path,
        comparison_path=comparison_path,
    )


class FakeRAGQueryService:
    """Return deterministic RAG answers without loading ML models."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                RetrievalStrategy | None,
            ]
        ] = []

    def ask(
        self,
        question: str,
        strategy: RetrievalStrategy | None = None,
    ) -> RAGQueryExecutionResult:
        self.calls.append(
            (
                question,
                strategy,
            )
        )

        selected_strategy = (
            strategy if strategy is not None else RetrievalStrategy.VECTOR
        )

        chunk = Chunk(
            chunk_id="inflation_basics::0",
            document_id="inflation_basics",
            content=(
                "Inflation describes a sustained increase in the general price level."
            ),
            source=("reference_corpus/inflation_basics"),
            metadata={
                "title": "Inflation Basics",
            },
        )

        answer = RAGAnswer(
            question=question,
            answer=("Inflation is a sustained increase in the general price level."),
            citations=(
                Citation(
                    document_id=("inflation_basics"),
                    source=("reference_corpus/inflation_basics"),
                    title=("Inflation Basics"),
                ),
            ),
            retrieved_chunks=(
                ScoredChunk(
                    chunk=chunk,
                    score=0.95,
                    rank=1,
                ),
            ),
        )

        return RAGQueryExecutionResult(
            strategy=selected_strategy,
            answer=answer,
        )


class FailingRAGQueryService:
    """Simulate an unavailable local RAG runtime."""

    def ask(
        self,
        question: str,
        strategy: RetrievalStrategy | None = None,
    ) -> RAGQueryExecutionResult:
        del question
        del strategy

        raise RAGQueryRuntimeError("RAG query execution failed.")


@pytest.fixture
def rag_client(
    tmp_path: Path,
) -> Iterator[
    tuple[
        TestClient,
        FakeRAGQueryService,
    ]
]:
    application = create_app()

    repository = _repository(tmp_path)
    service = FakeRAGQueryService()

    application.dependency_overrides[get_rag_repository] = lambda: repository

    application.dependency_overrides[get_rag_query_service] = lambda: service

    with TestClient(application) as client:
        yield client, service


def test_status_returns_rag_artifact_metadata(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, _ = rag_client

    response = client.get("/api/v1/rag/status")

    assert response.status_code == 200

    assert response.json() == {
        "corpus_available": True,
        "corpus_document_count": 2,
        "golden_dataset_available": True,
        "golden_dataset_item_count": 2,
        "benchmark_available": True,
        "default_strategy": "vector",
        "language_model": "gemma3:4b",
        "retrieval_strategies": [
            "vector",
            "hybrid",
            "reranked",
        ],
    }


def test_benchmark_exposes_quality_tie(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, _ = rag_client

    response = client.get("/api/v1/rag/benchmark")

    payload = response.json()

    assert response.status_code == 200
    assert payload["best_configuration"] == "vector"
    assert payload["quality_tie_for_first"] is True
    assert payload["generation_metrics_available"] is False
    assert len(payload["ranking"]) == 3


def test_query_uses_benchmark_default_when_strategy_is_omitted(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, service = rag_client

    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "What is inflation?",
        },
    )

    payload = response.json()

    assert response.status_code == 200
    assert payload["retrieval_strategy"] == "vector"
    assert payload["retrieved_chunk_count"] == 1

    assert payload["citations"] == [
        {
            "document_id": "inflation_basics",
            "source": ("reference_corpus/inflation_basics"),
            "title": "Inflation Basics",
            "page": None,
        }
    ]

    assert service.calls == [
        (
            "What is inflation?",
            None,
        )
    ]


def test_query_accepts_explicit_retrieval_strategy(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, service = rag_client

    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": ("What is financial risk?"),
            "strategy": "hybrid",
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieval_strategy"] == "hybrid"

    assert service.calls[-1] == (
        "What is financial risk?",
        RetrievalStrategy.HYBRID,
    )


def test_query_rejects_blank_question(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, _ = rag_client

    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "   ",
        },
    )

    assert response.status_code == 422


def test_query_rejects_unknown_strategy(
    rag_client: tuple[
        TestClient,
        FakeRAGQueryService,
    ],
) -> None:
    client, _ = rag_client

    response = client.post(
        "/api/v1/rag/query",
        json={
            "question": "What is inflation?",
            "strategy": "unknown",
        },
    )

    assert response.status_code == 422


def test_benchmark_returns_service_unavailable_when_comparison_is_missing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = RAGArtifactRepository(
        corpus_path=(tmp_path / "reference_corpus.json"),
        golden_dataset_path=(tmp_path / "golden_dataset_expanded.json"),
        comparison_path=(tmp_path / "comparison.json"),
    )

    application.dependency_overrides[get_rag_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/rag/benchmark")

    assert response.status_code == 503
    assert "RAG benchmark comparison is missing" in response.json()["detail"]


def test_status_reports_missing_optional_artifacts_without_failing(
    tmp_path: Path,
) -> None:
    application = create_app()

    repository = RAGArtifactRepository(
        corpus_path=(tmp_path / "reference_corpus.json"),
        golden_dataset_path=(tmp_path / "golden_dataset_expanded.json"),
        comparison_path=(tmp_path / "comparison.json"),
    )

    application.dependency_overrides[get_rag_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/rag/status")

    assert response.status_code == 200
    assert response.json() == {
        "corpus_available": False,
        "corpus_document_count": None,
        "golden_dataset_available": False,
        "golden_dataset_item_count": None,
        "benchmark_available": False,
        "default_strategy": None,
        "language_model": "gemma3:4b",
        "retrieval_strategies": [
            "vector",
            "hybrid",
            "reranked",
        ],
    }


def test_query_returns_service_unavailable_when_runtime_fails(
    tmp_path: Path,
) -> None:
    application = create_app()

    application.dependency_overrides[get_rag_query_service] = lambda: (
        FailingRAGQueryService()
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/rag/query",
            json={
                "question": "What is inflation?",
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": ("RAG query execution failed.")}


def test_benchmark_rejects_invalid_ranking_order(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    comparison_path = repository.comparison_path

    payload = json.loads(
        comparison_path.read_text(
            encoding="utf-8",
        )
    )

    payload["ranking"][0]["quality_score"] = 0.7
    payload["ranking"][1]["quality_score"] = 0.8

    comparison_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    application = create_app()

    application.dependency_overrides[get_rag_repository] = lambda: repository

    with TestClient(application) as client:
        response = client.get("/api/v1/rag/benchmark")

    assert response.status_code == 503
    assert "invalid schema" in response.json()["detail"]

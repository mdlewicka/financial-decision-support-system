"""Retrieval-augmented generation API routes."""

from functools import lru_cache
from math import isclose
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.src.api.config import (
    API_V1_PREFIX,
)
from backend.src.api.rag_repository import (
    RAGArtifactError,
    RAGArtifactRepository,
)
from backend.src.api.rag_schemas import (
    RAGBenchmarkResponse,
    RAGCitationResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatusResponse,
)
from backend.src.api.rag_service import (
    RAGQueryRuntimeError,
    RAGQueryService,
)
from backend.src.rag.config import (
    RAGConfig,
    RetrievalStrategy,
)

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/rag",
    tags=["rag"],
)


def get_rag_repository() -> RAGArtifactRepository:
    """Create the repository used to read frozen RAG artifacts."""
    return RAGArtifactRepository()


@lru_cache(maxsize=1)
def get_rag_query_service() -> RAGQueryService:
    """Create the process-level lazy RAG query service."""
    return RAGQueryService(
        repository=RAGArtifactRepository(),
    )


RAGRepositoryDependency = Annotated[
    RAGArtifactRepository,
    Depends(get_rag_repository),
]

RAGQueryServiceDependency = Annotated[
    RAGQueryService,
    Depends(get_rag_query_service),
]


@router.get(
    "/status",
    response_model=RAGStatusResponse,
    summary="Check RAG data and benchmark availability",
)
def get_rag_status(
    repository: RAGRepositoryDependency,
) -> RAGStatusResponse:
    """Return availability and metadata without initializing ML models."""
    corpus_available = repository.corpus_exists()
    golden_dataset_available = repository.golden_dataset_exists()
    benchmark_available = repository.comparison_exists()

    try:
        corpus_document_count = (
            repository.count_corpus_documents() if corpus_available else None
        )

        golden_dataset_item_count = (
            repository.count_golden_dataset_items()
            if golden_dataset_available
            else None
        )

        default_strategy = (
            repository.load_comparison().best_configuration
            if benchmark_available
            else None
        )
    except RAGArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return RAGStatusResponse(
        corpus_available=corpus_available,
        corpus_document_count=(corpus_document_count),
        golden_dataset_available=(golden_dataset_available),
        golden_dataset_item_count=(golden_dataset_item_count),
        benchmark_available=(benchmark_available),
        default_strategy=(default_strategy),
        language_model=(RAGConfig().language_model),
        retrieval_strategies=tuple(RetrievalStrategy),
    )


@router.get(
    "/benchmark",
    response_model=RAGBenchmarkResponse,
    summary="Get RAG retrieval benchmark comparison",
)
def get_rag_benchmark(
    repository: RAGRepositoryDependency,
) -> RAGBenchmarkResponse:
    """Return frozen comparison of vector, hybrid and reranked retrieval."""
    try:
        comparison = repository.load_comparison()
    except RAGArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    quality_tie_for_first = False

    if len(comparison.ranking) > 1:
        quality_tie_for_first = isclose(
            comparison.ranking[0].quality_score,
            comparison.ranking[1].quality_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    generation_metrics_available = all(
        (entry.mean_faithfulness is not None and entry.mean_context_recall is not None)
        for entry in comparison.ranking
    )

    return RAGBenchmarkResponse(
        best_configuration=(comparison.best_configuration),
        quality_tie_for_first=(quality_tie_for_first),
        generation_metrics_available=(generation_metrics_available),
        ranking=comparison.ranking,
    )


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Ask a question using retrieval-augmented generation",
)
def query_rag(
    request: RAGQueryRequest,
    service: RAGQueryServiceDependency,
) -> RAGQueryResponse:
    """Generate an answer grounded in the local reference corpus."""
    try:
        execution = service.ask(
            question=request.question,
            strategy=request.strategy,
        )
    except (
        RAGArtifactError,
        RAGQueryRuntimeError,
    ) as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    answer = execution.answer

    return RAGQueryResponse(
        question=answer.question,
        answer=answer.answer,
        retrieval_strategy=(execution.strategy),
        citations=tuple(
            RAGCitationResponse(
                document_id=(citation.document_id),
                source=citation.source,
                title=citation.title,
                page=citation.page,
            )
            for citation in (answer.citations)
        ),
        retrieved_chunk_count=len(answer.retrieved_chunks),
    )

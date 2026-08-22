"""Lazy runtime service for retrieval-augmented generation."""

from dataclasses import dataclass, replace
from threading import Lock

from backend.src.api.rag_repository import (
    RAGArtifactRepository,
)
from backend.src.rag.config import (
    RAGConfig,
    RetrievalStrategy,
)
from backend.src.rag.factory import (
    build_rag_pipeline,
)
from backend.src.rag.models import RAGAnswer
from backend.src.rag.pipeline import RAGPipeline


class RAGQueryRuntimeError(RuntimeError):
    """Raised when the local RAG runtime cannot execute a query."""


@dataclass(frozen=True)
class RAGQueryExecutionResult:
    """Store a generated answer together with the strategy that produced it."""

    strategy: RetrievalStrategy
    answer: RAGAnswer


class RAGQueryService:
    """Create and cache RAG pipelines lazily by retrieval strategy."""

    def __init__(
        self,
        repository: RAGArtifactRepository,
        base_config: RAGConfig | None = None,
    ) -> None:
        self._repository = repository
        self._base_config = base_config if base_config is not None else RAGConfig()
        self._pipelines: dict[
            RetrievalStrategy,
            RAGPipeline,
        ] = {}
        self._pipeline_lock = Lock()

    @property
    def language_model(self) -> str:
        """Return the configured local language model name."""
        return self._base_config.language_model

    def ask(
        self,
        question: str,
        strategy: RetrievalStrategy | None = None,
    ) -> RAGQueryExecutionResult:
        """Answer one question using the selected or benchmark-default strategy."""
        selected_strategy = (
            strategy
            if strategy is not None
            else (self._repository.load_comparison().best_configuration)
        )

        pipeline = self._get_pipeline(selected_strategy)

        try:
            answer = pipeline.ask(question)
        except Exception as error:
            raise RAGQueryRuntimeError(
                "RAG query execution failed. "
                "Ensure that the local language model service is available."
            ) from error

        return RAGQueryExecutionResult(
            strategy=selected_strategy,
            answer=answer,
        )

    def _get_pipeline(
        self,
        strategy: RetrievalStrategy,
    ) -> RAGPipeline:
        cached = self._pipelines.get(strategy)

        if cached is not None:
            return cached

        with self._pipeline_lock:
            cached = self._pipelines.get(strategy)

            if cached is not None:
                return cached

            documents = self._repository.load_documents()

            config = replace(
                self._base_config,
                strategy=strategy,
            )

            try:
                pipeline = build_rag_pipeline(config)
                pipeline.index(documents)
            except Exception as error:
                raise RAGQueryRuntimeError(
                    "RAG runtime initialization failed. "
                    "Ensure that local embedding, reranking and "
                    "language-model dependencies are available."
                ) from error

            self._pipelines[strategy] = pipeline

            return pipeline

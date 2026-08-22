"""Schemas for retrieval-augmented generation exposed by the HTTP API."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.src.rag.config import RetrievalStrategy


class RAGBenchmarkRankingEntry(BaseModel):
    """Store benchmark metrics for one retrieval configuration."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    rank: int = Field(gt=0)
    configuration_name: RetrievalStrategy
    quality_score: float = Field(ge=0.0, le=1.0)
    item_count: int = Field(gt=0)

    mean_recall_at_k: float = Field(ge=0.0, le=1.0)
    mean_precision_at_k: float = Field(ge=0.0, le=1.0)
    mean_hit_rate: float = Field(ge=0.0, le=1.0)
    mean_reciprocal_rank: float = Field(ge=0.0, le=1.0)

    mean_faithfulness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    mean_context_recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    duration_seconds: float = Field(ge=0.0)


class RAGComparisonArtifact(BaseModel):
    """Validate the exported RAG configuration comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    best_configuration: RetrievalStrategy
    ranking: tuple[
        RAGBenchmarkRankingEntry,
        ...,
    ]

    @model_validator(mode="after")
    def validate_ranking(
        self,
    ) -> "RAGComparisonArtifact":
        if not self.ranking:
            raise ValueError(
                "RAG comparison ranking must not be empty.",
            )

        expected_ranks = tuple(
            range(
                1,
                len(self.ranking) + 1,
            )
        )
        actual_ranks = tuple(entry.rank for entry in self.ranking)

        if actual_ranks != expected_ranks:
            raise ValueError(
                "RAG comparison ranks must be consecutive and start at one.",
            )

        configurations = tuple(entry.configuration_name for entry in self.ranking)

        if len(configurations) != len(set(configurations)):
            raise ValueError(
                "RAG comparison configurations must be unique.",
            )

        if self.ranking[0].configuration_name != self.best_configuration:
            raise ValueError(
                "RAG best configuration must match the first ranking entry.",
            )

        quality_scores = tuple(entry.quality_score for entry in self.ranking)

        if quality_scores != tuple(
            sorted(
                quality_scores,
                reverse=True,
            )
        ):
            raise ValueError(
                "RAG comparison must be ordered by descending quality score.",
            )

        return self


class RAGStatusResponse(BaseModel):
    """Describe availability of RAG data and benchmark artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    corpus_available: bool
    corpus_document_count: int | None = Field(
        default=None,
        ge=0,
    )

    golden_dataset_available: bool
    golden_dataset_item_count: int | None = Field(
        default=None,
        ge=0,
    )

    benchmark_available: bool
    default_strategy: RetrievalStrategy | None

    language_model: str
    retrieval_strategies: tuple[
        RetrievalStrategy,
        ...,
    ]


class RAGBenchmarkResponse(BaseModel):
    """Expose the frozen RAG benchmark comparison."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    best_configuration: RetrievalStrategy
    quality_tie_for_first: bool
    generation_metrics_available: bool
    ranking: tuple[
        RAGBenchmarkRankingEntry,
        ...,
    ]


class RAGQueryRequest(BaseModel):
    """Define one retrieval-augmented generation request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )
    strategy: RetrievalStrategy | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(
        cls,
        value: str,
    ) -> str:
        """Strip surrounding whitespace from the question."""
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "question must not be empty.",
            )

        return normalized


class RAGCitationResponse(BaseModel):
    """Expose one source citation supporting a generated answer."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_id: str
    source: str
    title: str | None
    page: int | None = Field(
        default=None,
        ge=0,
    )


class RAGQueryResponse(BaseModel):
    """Expose a generated answer with citations and retrieval metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str
    answer: str
    retrieval_strategy: RetrievalStrategy
    citations: tuple[
        RAGCitationResponse,
        ...,
    ]
    retrieved_chunk_count: int = Field(
        ge=0,
    )


class RAGArtifactStatus(BaseModel):
    """Validate a small JSON artifact containing a list."""

    model_config = ConfigDict(
        extra="allow",
        frozen=True,
    )

    items: list[object]


RAGBenchmarkGenerationAvailability = Literal[
    "available",
    "not_evaluated",
]

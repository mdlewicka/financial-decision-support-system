"""Schemas for decision-support agent results exposed by the HTTP API."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from backend.src.api.decision_schemas import (
    DecisionEvaluationRequest,
    DecisionEvaluationResponse,
)
from backend.src.rag.config import RetrievalStrategy

AgentRiskPreference = Literal[
    "defensive",
    "balanced",
    "growth",
]
AgentResponseLanguage = Literal["pl", "en"]


class AgentEvidenceResponse(BaseModel):
    """Expose one evidence reference used by a decision-support agent."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_type: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    description: str = Field(min_length=1)


class AgentResponsePayload(BaseModel):
    """Expose one structured decision-support agent response."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    agent_name: str = Field(min_length=1)
    question: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    findings: tuple[str, ...] = Field(min_length=1)
    risks: tuple[str, ...] = Field(min_length=1)
    recommendations: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    evidence: tuple[AgentEvidenceResponse, ...] = Field(min_length=1)

    @field_validator(
        "findings",
        "risks",
        "recommendations",
        "limitations",
    )
    @classmethod
    def validate_text_items(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require non-empty text in every structured response item."""
        if any(not value.strip() for value in values):
            raise ValueError(
                "Agent response text items must not be empty.",
            )

        return values


class AgentResponsesArtifact(BaseModel):
    """Store the three frozen decision-support agent responses."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    forecast: AgentResponsePayload
    risk: AgentResponsePayload
    allocation: AgentResponsePayload

    @model_validator(mode="after")
    def validate_agent_roles(
        self,
    ) -> "AgentResponsesArtifact":
        expected_names = {
            "forecast": "forecast_analysis_agent",
            "risk": "risk_analysis_agent",
            "allocation": "allocation_recommendation_agent",
        }

        for role, expected_name in expected_names.items():
            response = getattr(
                self,
                role,
            )

            if response.agent_name != expected_name:
                raise ValueError(
                    f"Frozen {role} response must come from {expected_name!r}.",
                )

        return self


class AgentResultsArtifact(BaseModel):
    """Validate the exported decision-support agent workflow artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1]
    model_name: str = Field(min_length=1)
    retrieval_strategy: RetrievalStrategy
    evaluation_basis: Literal["validation_benchmark_artifacts"]
    test_split_used: Literal[False]
    responses: AgentResponsesArtifact


class AgentStatusResponse(BaseModel):
    """Describe availability of agent results and runtime dependencies."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    result_artifact_available: bool
    forecast_artifact_available: bool
    xai_artifact_available: bool
    portfolio_artifact_available: bool
    reinforcement_learning_artifact_available: bool
    corpus_available: bool
    runtime_inputs_available: bool
    model_name: str | None
    retrieval_strategy: RetrievalStrategy | None
    evaluation_basis: Literal["validation_benchmark_artifacts"] | None
    test_split_used: bool | None


class AgentResultsResponse(BaseModel):
    """Expose the frozen three-agent validation workflow."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    retrieval_strategy: RetrievalStrategy
    evaluation_basis: Literal["validation_benchmark_artifacts"]
    test_split_used: Literal[False]
    responses: AgentResponsesArtifact


class AgentQuestionRequest(BaseModel):
    """Define a free-text question for a decision-support agent."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    @field_validator("question")
    @classmethod
    def normalize_question(
        cls,
        value: str,
    ) -> str:
        """Strip surrounding whitespace from the user question."""
        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "question must not be empty.",
            )

        return normalized


class ForecastAgentRequest(AgentQuestionRequest):
    """Define an interactive forecast-agent request."""

    ticker: str | None = Field(
        default=None,
        min_length=1,
        max_length=32,
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize an optional asset ticker."""
        if value is None:
            return None

        normalized = value.strip().upper()

        if not normalized:
            raise ValueError(
                "ticker must not be empty.",
            )

        return normalized


class RiskAgentRequest(AgentQuestionRequest):
    """Define an interactive risk-agent request."""


class AllocationAgentRequest(AgentQuestionRequest):
    """Define an interactive allocation-agent request."""

    risk_preference: AgentRiskPreference = "balanced"


class DecisionNarrativeRequest(AgentQuestionRequest):
    """Request a narrative for a server-recomputed Decision Engine result."""

    evaluation: DecisionEvaluationRequest
    response_language: AgentResponseLanguage = "pl"


class DecisionNarrativeResponse(BaseModel):
    """Bind an agent narrative to the deterministic result it explains."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    decision: DecisionEvaluationResponse
    narrative: AgentResponsePayload

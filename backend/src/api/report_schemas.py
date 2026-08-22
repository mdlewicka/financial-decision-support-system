"""HTTP schemas for auditable decision-report generation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.src.api.agent_schemas import (
    AgentQuestionRequest,
    AgentResponseLanguage,
)
from backend.src.api.decision_schemas import DecisionEvaluationRequest


class DecisionReportRequest(AgentQuestionRequest):
    """Recompute a decision and create a report from trusted server evidence."""

    evaluation: DecisionEvaluationRequest
    response_language: AgentResponseLanguage = "pl"
    presentation_timezone: str = Field(
        default="Europe/Warsaw",
        min_length=1,
        max_length=64,
    )

    @field_validator("presentation_timezone")
    @classmethod
    def normalize_timezone(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("presentation_timezone must not be empty.")
        return normalized


class DecisionReportFileResponse(BaseModel):
    """One integrity-protected file embedded in the JSON response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["json", "markdown", "pdf"]
    file_name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(min_length=1)


class DecisionReportResponse(BaseModel):
    """Report data, exact downloadable files and their integrity manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    report: dict[str, Any]
    manifest: dict[str, Any]
    files: tuple[DecisionReportFileResponse, ...] = Field(min_length=3, max_length=3)

"""Auditable decision-report API routes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, status

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.report_schemas import (
    DecisionReportFileResponse,
    DecisionReportRequest,
    DecisionReportResponse,
)
from backend.src.api.report_service import (
    DecisionReportGenerationError,
    DecisionReportRuntimeService,
)
from backend.src.api.routes.agents import (
    AgentRepositoryDependency,
    AgentRuntimeDependency,
)
from backend.src.api.routes.decision import (
    DecisionPortfolioRepositoryDependency,
    evaluate_decision_request,
)

router = APIRouter(prefix=f"{API_V1_PREFIX}/reports", tags=["reports"])

ReportFormat = Literal["json", "markdown", "pdf"]


def get_decision_report_service(
    agent_service: AgentRuntimeDependency,
    agent_repository: AgentRepositoryDependency,
) -> DecisionReportRuntimeService:
    """Create the request-independent report orchestration service."""
    return DecisionReportRuntimeService(
        agent_service=agent_service,
        agent_repository=agent_repository,
    )


def get_report_generated_at() -> datetime:
    """Return the report timestamp through an overridable dependency."""
    return datetime.now(UTC)


DecisionReportServiceDependency = Annotated[
    DecisionReportRuntimeService,
    Depends(get_decision_report_service),
]
ReportGeneratedAtDependency = Annotated[
    datetime,
    Depends(get_report_generated_at),
]


@router.post(
    "/decision",
    response_model=DecisionReportResponse,
    summary="Generate an auditable report for a server-recomputed decision",
)
def generate_decision_report(
    request: DecisionReportRequest,
    report_service: DecisionReportServiceDependency,
    portfolio_repository: DecisionPortfolioRepositoryDependency,
    generated_at: ReportGeneratedAtDependency,
) -> DecisionReportResponse:
    """Return exact JSON, Markdown and PDF files without trusting client results."""
    decision = evaluate_decision_request(
        request.evaluation,
        portfolio_repository,
    )

    try:
        bundle = report_service.generate_bundle(
            decision=decision.model_dump(mode="json"),
            question=request.question,
            presentation_timezone=request.presentation_timezone,
            generated_at=generated_at,
            response_language=request.response_language,
        )
    except DecisionReportGenerationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return DecisionReportResponse(
        report=bundle.report,
        manifest=bundle.manifest,
        files=tuple(
            DecisionReportFileResponse(
                format=cast(ReportFormat, item.format),
                file_name=item.file_name,
                media_type=item.media_type,
                bytes=item.bytes,
                sha256=item.sha256,
                content_base64=item.content_base64,
            )
            for item in bundle.files
        ),
    )

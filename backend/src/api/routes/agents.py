"""Decision-support agent API routes."""

from functools import lru_cache
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.src.agents.models import (
    AgentResponse,
)
from backend.src.api.agent_repository import (
    AgentArtifactError,
    AgentResultRepository,
)
from backend.src.api.agent_schemas import (
    AgentEvidenceResponse,
    AgentResponsePayload,
    AgentResultsResponse,
    AgentStatusResponse,
    AllocationAgentRequest,
    DecisionNarrativeRequest,
    DecisionNarrativeResponse,
    ForecastAgentRequest,
    RiskAgentRequest,
)
from backend.src.api.agent_service import (
    AgentRuntimeError,
    AgentRuntimeService,
)
from backend.src.api.config import API_V1_PREFIX
from backend.src.api.routes.decision import (
    DecisionPortfolioRepositoryDependency,
    evaluate_decision_request,
)
from backend.src.config import SUPPORTED_TICKERS

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/agents",
    tags=["agents"],
)


def get_agent_repository() -> AgentResultRepository:
    """Create the repository used to read frozen agent artifacts."""
    return AgentResultRepository()


@lru_cache(maxsize=1)
def get_agent_runtime_service() -> AgentRuntimeService:
    """Create the process-level lazy interactive agent service."""
    return AgentRuntimeService(
        repository=AgentResultRepository(),
    )


AgentRepositoryDependency = Annotated[
    AgentResultRepository,
    Depends(get_agent_repository),
]

AgentRuntimeDependency = Annotated[
    AgentRuntimeService,
    Depends(get_agent_runtime_service),
]


@router.get(
    "/status",
    response_model=AgentStatusResponse,
    summary="Check agent artifacts and runtime input availability",
)
def get_agent_status(
    repository: AgentRepositoryDependency,
) -> AgentStatusResponse:
    """Return agent artifact availability without loading ML models."""
    result_available = repository.result_exists()

    result_metadata = None

    if result_available:
        try:
            result_metadata = repository.load_results()
        except AgentArtifactError as error:
            raise HTTPException(
                status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
                detail=str(error),
            ) from error

    return AgentStatusResponse(
        result_artifact_available=(result_available),
        forecast_artifact_available=(repository.forecast_artifact_exists()),
        xai_artifact_available=(repository.xai_artifact_exists()),
        portfolio_artifact_available=(repository.portfolio_artifact_exists()),
        reinforcement_learning_artifact_available=(
            repository.reinforcement_learning_artifact_exists()
        ),
        corpus_available=(repository.corpus_exists()),
        runtime_inputs_available=(repository.runtime_inputs_available()),
        model_name=(
            result_metadata.model_name if result_metadata is not None else None
        ),
        retrieval_strategy=(
            result_metadata.retrieval_strategy if result_metadata is not None else None
        ),
        evaluation_basis=(
            result_metadata.evaluation_basis if result_metadata is not None else None
        ),
        test_split_used=(
            result_metadata.test_split_used if result_metadata is not None else None
        ),
    )


@router.get(
    "/results",
    response_model=AgentResultsResponse,
    summary="Get frozen decision-support agent results",
)
def get_agent_results(
    repository: AgentRepositoryDependency,
) -> AgentResultsResponse:
    """Return the reproducible three-agent validation workflow result."""
    try:
        artifact = repository.load_results()
    except AgentArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return AgentResultsResponse(
        model_name=artifact.model_name,
        retrieval_strategy=(artifact.retrieval_strategy),
        evaluation_basis=(artifact.evaluation_basis),
        test_split_used=(artifact.test_split_used),
        responses=artifact.responses,
    )


@router.post(
    "/forecast",
    response_model=AgentResponsePayload,
    summary="Run the forecast-analysis agent",
)
def run_forecast_agent(
    request: ForecastAgentRequest,
    service: AgentRuntimeDependency,
) -> AgentResponsePayload:
    """Interpret forecast and XAI validation evidence."""
    if request.ticker is not None and request.ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=(f"Unsupported ticker: {request.ticker!r}."),
        )

    try:
        response = service.analyze_forecast(
            question=request.question,
            ticker=request.ticker,
        )
    except AgentRuntimeError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return _to_response_payload(response)


@router.post(
    "/risk",
    response_model=AgentResponsePayload,
    summary="Run the risk-analysis agent",
)
def run_risk_agent(
    request: RiskAgentRequest,
    service: AgentRuntimeDependency,
) -> AgentResponsePayload:
    """Compare validated portfolio risk scenarios."""
    try:
        response = service.analyze_risk(
            question=request.question,
        )
    except AgentRuntimeError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return _to_response_payload(response)


@router.post(
    "/allocation",
    response_model=AgentResponsePayload,
    summary="Run the allocation-recommendation agent",
)
def run_allocation_agent(
    request: AllocationAgentRequest,
    service: AgentRuntimeDependency,
) -> AgentResponsePayload:
    """Compare evaluated allocations for a declared risk preference."""
    try:
        response = service.recommend_allocation(
            question=request.question,
            risk_preference=(request.risk_preference),
        )
    except AgentRuntimeError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return _to_response_payload(response)


@router.post(
    "/decision",
    response_model=DecisionNarrativeResponse,
    summary="Explain a server-recomputed Decision Engine result",
)
def run_decision_narrative_agent(
    request: DecisionNarrativeRequest,
    service: AgentRuntimeDependency,
    repository: DecisionPortfolioRepositoryDependency,
) -> DecisionNarrativeResponse:
    """Explain verified validation evidence without trusting client-side results."""
    decision = evaluate_decision_request(
        request.evaluation,
        repository,
    )

    try:
        response = service.explain_decision(
            question=request.question,
            decision_context=decision.model_dump(mode="json"),
            response_language=request.response_language,
        )
    except AgentRuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return DecisionNarrativeResponse(
        decision=decision,
        narrative=_to_response_payload(response),
    )


def _to_response_payload(
    response: AgentResponse,
) -> AgentResponsePayload:
    return AgentResponsePayload(
        agent_name=response.agent_name,
        question=response.question,
        summary=response.summary,
        findings=response.findings,
        risks=response.risks,
        recommendations=response.recommendations,
        limitations=response.limitations,
        evidence=tuple(
            AgentEvidenceResponse(
                source_type=(reference.source_type),
                identifier=(reference.identifier),
                description=(reference.description),
            )
            for reference in response.evidence
        ),
    )

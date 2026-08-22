"""Portfolio risk benchmark API routes."""

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
from backend.src.api.risk_repository import (
    RiskArtifactError,
    RiskArtifactRepository,
)
from backend.src.api.risk_schemas import (
    RiskArtifactPayload,
    RiskMarketResponse,
    RiskScenarioDetailResponse,
    RiskScenariosResponse,
    RiskScenarioSummary,
    RiskStatusResponse,
)

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/risk",
    tags=["risk"],
)


def get_risk_repository() -> RiskArtifactRepository:
    """Create the repository used to read frozen risk benchmark artifacts."""
    return RiskArtifactRepository()


RiskRepositoryDependency = Annotated[
    RiskArtifactRepository,
    Depends(get_risk_repository),
]


@router.get(
    "/status",
    response_model=RiskStatusResponse,
    summary="Check risk benchmark artifact availability",
)
def get_risk_status(
    repository: RiskRepositoryDependency,
) -> RiskStatusResponse:
    """Return availability of frozen portfolio risk artifacts."""
    return RiskStatusResponse(
        artifact_available=(repository.artifact_exists()),
        validation_correlation_available=(repository.validation_correlation_exists()),
    )


@router.get(
    "/market",
    response_model=RiskMarketResponse,
    summary="Get validation market risk context",
)
def get_risk_market(
    repository: RiskRepositoryDependency,
) -> RiskMarketResponse:
    """Return validation market metadata and correlations."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    validation_split = artifact.market_splits.validation

    try:
        correlation_matrix = repository.load_validation_correlation(
            validation_split.tickers,
        )
    except RiskArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return RiskMarketResponse(
        evaluation_split="validation",
        return_definition=(artifact.return_definition),
        observations=(validation_split.observations),
        periods_per_year=(validation_split.periods_per_year),
        date_min=(validation_split.date_min),
        date_max=(validation_split.date_max),
        tickers=(validation_split.tickers),
        correlation_matrix=(correlation_matrix),
    )


@router.get(
    "/scenarios",
    response_model=RiskScenariosResponse,
    summary="Compare validation risk across portfolio scenarios",
)
def get_risk_scenarios(
    repository: RiskRepositoryDependency,
) -> RiskScenariosResponse:
    """Return validation risk measures for every portfolio scenario."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    return RiskScenariosResponse(
        evaluation_split="validation",
        return_definition=(artifact.return_definition),
        scenarios=tuple(
            RiskScenarioSummary(
                name=scenario.name,
                construction=(scenario.construction),
                risk=(scenario.validation),
            )
            for scenario in (artifact.scenarios)
        ),
    )


@router.get(
    "/scenarios/{scenario_name}",
    response_model=RiskScenarioDetailResponse,
    summary="Get risk details for one portfolio scenario",
)
def get_risk_scenario(
    scenario_name: str,
    repository: RiskRepositoryDependency,
) -> RiskScenarioDetailResponse:
    """Return training and validation risk for one named scenario."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    normalized_name = scenario_name.strip().lower()

    scenario = next(
        (
            candidate
            for candidate in (artifact.scenarios)
            if (candidate.name.lower() == normalized_name)
        ),
        None,
    )

    if scenario is None:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=(f"Risk scenario is not available: {scenario_name!r}."),
        )

    return RiskScenarioDetailResponse(
        name=scenario.name,
        construction=(scenario.construction),
        return_definition=(artifact.return_definition),
        training=(scenario.training),
        validation=(scenario.validation),
    )


def _load_or_service_unavailable(
    repository: RiskArtifactRepository,
) -> RiskArtifactPayload:
    try:
        return repository.load()
    except RiskArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

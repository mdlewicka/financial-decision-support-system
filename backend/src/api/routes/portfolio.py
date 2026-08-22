"""Portfolio benchmark API routes."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.portfolio_repository import (
    PortfolioArtifactError,
    PortfolioArtifactRepository,
)
from backend.src.api.portfolio_schemas import (
    PortfolioPerformance,
    PortfolioReturnsResponse,
    PortfolioScenarioDetailResponse,
    PortfolioScenariosResponse,
    PortfolioScenarioSummary,
    PortfolioStatusResponse,
    PortfolioWeight,
)
from backend.src.api.risk_schemas import (
    RiskAnalysisArtifact,
    RiskArtifactPayload,
    RiskScenarioArtifact,
)

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/portfolio",
    tags=["portfolio"],
)


def get_portfolio_repository() -> PortfolioArtifactRepository:
    """Create the repository used to read frozen portfolio artifacts."""
    return PortfolioArtifactRepository()


PortfolioRepositoryDependency = Annotated[
    PortfolioArtifactRepository,
    Depends(get_portfolio_repository),
]


@router.get(
    "/status",
    response_model=PortfolioStatusResponse,
    summary="Check portfolio artifact availability",
)
def get_portfolio_status(
    repository: PortfolioRepositoryDependency,
) -> PortfolioStatusResponse:
    """Return availability of frozen portfolio artifacts."""
    return PortfolioStatusResponse(
        artifact_available=(repository.artifact_exists()),
        contributions_available=(repository.contributions_exist()),
        returns_available=(repository.returns_exist()),
    )


@router.get(
    "/scenarios",
    response_model=PortfolioScenariosResponse,
    summary="Compare portfolio allocation scenarios",
)
def get_portfolio_scenarios(
    repository: PortfolioRepositoryDependency,
) -> PortfolioScenariosResponse:
    """Return weights and validation performance for all scenarios."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    return PortfolioScenariosResponse(
        evaluation_split="validation",
        return_definition=(artifact.return_definition),
        tickers=(artifact.market_splits.validation.tickers),
        scenarios=tuple(
            PortfolioScenarioSummary(
                name=scenario.name,
                construction=scenario.construction,
                weights=_to_weights(scenario),
                validation=_to_performance(scenario.validation),
            )
            for scenario in artifact.scenarios
        ),
    )


@router.get(
    "/scenarios/{scenario_name}",
    response_model=PortfolioScenarioDetailResponse,
    summary="Get one portfolio allocation scenario",
)
def get_portfolio_scenario(
    scenario_name: str,
    repository: PortfolioRepositoryDependency,
) -> PortfolioScenarioDetailResponse:
    """Return allocation, performance and contributions for one scenario."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    scenario = _find_scenario(
        artifact,
        scenario_name,
    )

    try:
        contributions = repository.load_validation_contributions(
            scenario_name=scenario.name,
            construction=scenario.construction,
            expected_weights=scenario.weights,
        )
    except PortfolioArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return PortfolioScenarioDetailResponse(
        name=scenario.name,
        construction=scenario.construction,
        return_definition=(artifact.return_definition),
        weights=_to_weights(scenario),
        training=_to_performance(scenario.training),
        validation=_to_performance(scenario.validation),
        validation_contributions=contributions,
    )


@router.get(
    "/scenarios/{scenario_name}/returns",
    response_model=PortfolioReturnsResponse,
    summary="Get validation returns for one portfolio scenario",
)
def get_portfolio_returns(
    scenario_name: str,
    repository: PortfolioRepositoryDependency,
) -> PortfolioReturnsResponse:
    """Return validation daily and cumulative returns for one scenario."""
    artifact = _load_or_service_unavailable(
        repository,
    )

    scenario = _find_scenario(
        artifact,
        scenario_name,
    )

    validation = artifact.market_splits.validation

    try:
        returns = repository.load_validation_returns(
            scenario_name=scenario.name,
            construction=scenario.construction,
            expected_observations=(validation.observations),
            expected_date_min=(validation.date_min),
            expected_date_max=(validation.date_max),
        )
    except PortfolioArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

    return PortfolioReturnsResponse(
        name=scenario.name,
        construction=scenario.construction,
        evaluation_split="validation",
        return_definition=(artifact.return_definition),
        observations=len(returns),
        returns=returns,
    )


def _find_scenario(
    artifact: RiskArtifactPayload,
    scenario_name: str,
) -> RiskScenarioArtifact:
    normalized_name = scenario_name.strip().lower()

    scenario = next(
        (
            candidate
            for candidate in artifact.scenarios
            if (candidate.name.lower() == normalized_name)
        ),
        None,
    )

    if scenario is None:
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=(f"Portfolio scenario is not available: {scenario_name!r}."),
        )

    return scenario


def _to_weights(
    scenario: RiskScenarioArtifact,
) -> tuple[PortfolioWeight, ...]:
    return tuple(
        PortfolioWeight(
            ticker=ticker,
            weight=weight,
        )
        for ticker, weight in (scenario.weights.items())
    )


def _to_performance(
    analysis: RiskAnalysisArtifact,
) -> PortfolioPerformance:
    metrics = analysis.metrics

    return PortfolioPerformance(
        observations=metrics.observations,
        annualized_mean_return=(metrics.annualized_mean_return),
        annualized_volatility=(metrics.annualized_volatility),
        sharpe_ratio=metrics.sharpe_ratio,
        sortino_ratio=metrics.sortino_ratio,
        maximum_drawdown=(metrics.maximum_drawdown),
        concentration_index=(analysis.concentration_index),
        effective_number_of_assets=(analysis.effective_number_of_assets),
        diversification_ratio=(analysis.diversification_ratio),
    )


def _load_or_service_unavailable(
    repository: PortfolioArtifactRepository,
) -> RiskArtifactPayload:
    try:
        return repository.load()
    except PortfolioArtifactError as error:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=str(error),
        ) from error

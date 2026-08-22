"""Forecast benchmark API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.forecast_repository import (
    ForecastArtifactError,
    ForecastArtifactRepository,
    ForecastModelDefinition,
)
from backend.src.api.forecast_schemas import (
    ForecastArtifactPayload,
    ForecastModelAvailability,
    ForecastModelsResponse,
    ForecastModelSummary,
    ForecastSummaryResponse,
    TickerForecastModelResult,
    TickerForecastResponse,
)
from backend.src.config import SUPPORTED_TICKERS

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/forecast",
    tags=["forecast"],
)


def get_forecast_repository() -> ForecastArtifactRepository:
    """Create the repository used to read frozen forecast artifacts."""
    return ForecastArtifactRepository()


ForecastRepositoryDependency = Annotated[
    ForecastArtifactRepository,
    Depends(get_forecast_repository),
]


@router.get(
    "/models",
    response_model=ForecastModelsResponse,
    summary="List forecast benchmark models",
)
def get_forecast_models(
    repository: ForecastRepositoryDependency,
) -> ForecastModelsResponse:
    """List forecast models and benchmark artifact availability."""
    return ForecastModelsResponse(
        models=tuple(
            ForecastModelAvailability(
                model_name=definition.model_name,
                display_name=definition.display_name,
                artifact_available=repository.artifact_exists(definition),
            )
            for definition in repository.model_definitions
        )
    )


@router.get(
    "/summary",
    response_model=ForecastSummaryResponse,
    summary="Compare forecast validation results",
)
def get_forecast_summary(
    repository: ForecastRepositoryDependency,
) -> ForecastSummaryResponse:
    """Return overall validation metrics for all forecast models."""
    artifacts = _load_all_or_service_unavailable(
        repository,
    )

    return ForecastSummaryResponse(
        evaluation_split="validation",
        target_definition="next_session_simple_return",
        tickers=SUPPORTED_TICKERS,
        models=tuple(
            ForecastModelSummary(
                model_name=definition.model_name,
                display_name=definition.display_name,
                metrics=artifact.metrics.overall,
            )
            for definition, artifact in artifacts
        ),
    )


@router.get(
    "/tickers/{ticker}",
    response_model=TickerForecastResponse,
    summary="Compare forecast validation results for one ticker",
)
def get_ticker_forecast(
    ticker: str,
    repository: ForecastRepositoryDependency,
) -> TickerForecastResponse:
    """Return per-ticker validation metrics for all forecast models."""
    normalized_ticker = ticker.strip().upper()

    if normalized_ticker not in SUPPORTED_TICKERS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(f"Unsupported ticker: {normalized_ticker or ticker!r}."),
        )

    artifacts = _load_all_or_service_unavailable(
        repository,
    )
    model_results: list[TickerForecastModelResult] = []

    for definition, artifact in artifacts:
        ticker_metrics = artifact.metrics.by_ticker.get(
            normalized_ticker,
        )

        if ticker_metrics is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Forecast benchmark artifact does not contain "
                    f"metrics for ticker {normalized_ticker!r} "
                    f"and model {definition.model_name!r}."
                ),
            )

        model_results.append(
            TickerForecastModelResult(
                model_name=definition.model_name,
                display_name=definition.display_name,
                metrics=ticker_metrics,
            )
        )

    return TickerForecastResponse(
        ticker=normalized_ticker,
        evaluation_split="validation",
        target_definition="next_session_simple_return",
        models=tuple(model_results),
    )


def _load_all_or_service_unavailable(
    repository: ForecastArtifactRepository,
) -> tuple[
    tuple[
        ForecastModelDefinition,
        ForecastArtifactPayload,
    ],
    ...,
]:
    try:
        return repository.load_all()
    except ForecastArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

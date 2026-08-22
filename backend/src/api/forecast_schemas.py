"""Forecast benchmark schemas exposed by the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ForecastEvaluationSplit = Literal["validation"]
ForecastTargetDefinition = Literal["next_session_simple_return"]


class ForecastMetricValues(BaseModel):
    """Store forecast evaluation metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    observations: int = Field(ge=0)
    mae: float | None
    mse: float | None
    rmse: float | None
    r2: float | None
    directional_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


class ForecastMetricsCollection(BaseModel):
    """Store overall and per-ticker forecast metrics."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    overall: ForecastMetricValues
    by_ticker: dict[str, ForecastMetricValues]


class ForecastArtifactPayload(BaseModel):
    """Validate the reusable part of an exported forecast benchmark."""

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    schema_version: Literal[1]
    model_name: str
    evaluation_split: ForecastEvaluationSplit
    target_definition: ForecastTargetDefinition
    metrics: ForecastMetricsCollection


class ForecastModelAvailability(BaseModel):
    """Describe whether a benchmark artifact is available."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    display_name: str
    artifact_available: bool


class ForecastModelsResponse(BaseModel):
    """List forecast models exposed by the application."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    models: tuple[ForecastModelAvailability, ...]


class ForecastModelSummary(BaseModel):
    """Store overall validation metrics for one forecast model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    display_name: str
    metrics: ForecastMetricValues


class ForecastSummaryResponse(BaseModel):
    """Compare validation metrics across forecast models."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evaluation_split: ForecastEvaluationSplit
    target_definition: ForecastTargetDefinition
    tickers: tuple[str, ...]
    models: tuple[ForecastModelSummary, ...]


class TickerForecastModelResult(BaseModel):
    """Store validation metrics for one model and ticker."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    model_name: str
    display_name: str
    metrics: ForecastMetricValues


class TickerForecastResponse(BaseModel):
    """Compare forecast models for one supported ticker."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    ticker: str
    evaluation_split: ForecastEvaluationSplit
    target_definition: ForecastTargetDefinition
    models: tuple[TickerForecastModelResult, ...]

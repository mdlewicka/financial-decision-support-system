"""Schemas for portfolio results exposed by the HTTP API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PortfolioEvaluationSplit = Literal["validation"]
PortfolioReturnDefinition = Literal[
    "adjusted_close_simple_return_on_shared_market_dates"
]


class PortfolioStatusResponse(BaseModel):
    """Describe availability of frozen portfolio artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_available: bool
    contributions_available: bool
    returns_available: bool


class PortfolioWeight(BaseModel):
    """Represent one asset weight in a portfolio."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    ticker: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)


class PortfolioPerformance(BaseModel):
    """Expose key performance and diversification measures."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    observations: int = Field(gt=0)
    annualized_mean_return: float
    annualized_volatility: float = Field(ge=0.0)
    sharpe_ratio: float | None
    sortino_ratio: float | None
    maximum_drawdown: float = Field(ge=0.0)
    concentration_index: float = Field(ge=0.0, le=1.0)
    effective_number_of_assets: float = Field(ge=1.0)
    diversification_ratio: float | None = Field(
        default=None,
        gt=0.0,
    )


class PortfolioScenarioSummary(BaseModel):
    """Expose one portfolio scenario in the comparison view."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    construction: str
    weights: tuple[PortfolioWeight, ...]
    validation: PortfolioPerformance


class PortfolioScenariosResponse(BaseModel):
    """Compare all frozen portfolio allocation scenarios."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evaluation_split: PortfolioEvaluationSplit
    return_definition: PortfolioReturnDefinition
    tickers: tuple[str, ...]
    scenarios: tuple[PortfolioScenarioSummary, ...]


class PortfolioContribution(BaseModel):
    """Expose one asset contribution to portfolio return and volatility."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    ticker: str
    weight: float = Field(ge=0.0, le=1.0)
    annualized_return_contribution: float
    annualized_volatility_contribution: float


class PortfolioScenarioDetailResponse(BaseModel):
    """Expose allocation, performance and contributions for one scenario."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    construction: str
    return_definition: PortfolioReturnDefinition
    weights: tuple[PortfolioWeight, ...]
    training: PortfolioPerformance
    validation: PortfolioPerformance
    validation_contributions: tuple[
        PortfolioContribution,
        ...,
    ]


class PortfolioReturnPoint(BaseModel):
    """Represent one validation portfolio return observation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    date: date
    portfolio_return: float = Field(ge=-1.0)
    cumulative_return: float


class PortfolioReturnsResponse(BaseModel):
    """Expose the validation portfolio return time series."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    construction: str
    evaluation_split: PortfolioEvaluationSplit
    return_definition: PortfolioReturnDefinition
    observations: int = Field(gt=0)
    returns: tuple[
        PortfolioReturnPoint,
        ...,
    ]

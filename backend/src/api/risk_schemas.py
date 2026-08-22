"""Schemas for portfolio risk results exposed by the HTTP API."""

from datetime import date
from math import isclose
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskEvaluationSplit = Literal["validation"]
RiskReturnDefinition = Literal["adjusted_close_simple_return_on_shared_market_dates"]


class RiskMetricValues(BaseModel):
    """Store risk metrics for one ordered portfolio return series."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    observations: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    annual_risk_free_rate: float
    mean_return: float
    annualized_mean_return: float
    volatility: float = Field(ge=0.0)
    annualized_volatility: float = Field(ge=0.0)
    sharpe_ratio: float | None
    sortino_ratio: float | None
    value_at_risk: float
    conditional_value_at_risk: float
    maximum_drawdown: float = Field(ge=0.0)


class RiskAnalysisArtifact(BaseModel):
    """Store risk and diversification measures for one portfolio split."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    metrics: RiskMetricValues
    concentration_index: float = Field(
        ge=0.0,
        le=1.0,
    )
    effective_number_of_assets: float = Field(
        ge=1.0,
    )
    diversification_ratio: float | None = Field(
        default=None,
        gt=0.0,
    )


class RiskMarketSplitArtifact(BaseModel):
    """Describe one market split used by the risk benchmark."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    observations: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    date_min: date
    date_max: date
    tickers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_split(
        self,
    ) -> "RiskMarketSplitArtifact":
        if self.date_max < self.date_min:
            raise ValueError(
                "Risk market split date_max must not precede date_min.",
            )

        normalized_tickers = tuple(ticker.strip() for ticker in self.tickers)

        if not normalized_tickers:
            raise ValueError(
                "Risk market split must contain at least one ticker.",
            )

        if any(not ticker for ticker in normalized_tickers):
            raise ValueError(
                "Risk market split tickers must not contain empty values.",
            )

        if len(normalized_tickers) != len(set(normalized_tickers)):
            raise ValueError(
                "Risk market split tickers must be unique.",
            )

        return self


class RiskMarketSplitsArtifact(BaseModel):
    """Store training and validation market split metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    training: RiskMarketSplitArtifact
    validation: RiskMarketSplitArtifact

    @model_validator(mode="after")
    def validate_consistent_market_universe(
        self,
    ) -> "RiskMarketSplitsArtifact":
        if self.training.tickers != self.validation.tickers:
            raise ValueError(
                "Training and validation risk market tickers must match.",
            )

        if self.training.periods_per_year != self.validation.periods_per_year:
            raise ValueError(
                "Training and validation periods_per_year must match.",
            )

        return self


class RiskScenarioArtifact(BaseModel):
    """Store one fixed portfolio scenario evaluated on both splits."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    name: str = Field(min_length=1)
    construction: str = Field(min_length=1)
    weights: dict[str, float]
    training: RiskAnalysisArtifact
    validation: RiskAnalysisArtifact

    @model_validator(mode="after")
    def validate_weights(
        self,
    ) -> "RiskScenarioArtifact":
        if not self.weights:
            raise ValueError(
                "Risk scenario weights must not be empty.",
            )

        if any(not ticker.strip() for ticker in self.weights):
            raise ValueError(
                "Risk scenario weight tickers must not be empty.",
            )

        if any(weight < 0.0 for weight in self.weights.values()):
            raise ValueError(
                "Risk scenario weights must not be negative.",
            )

        if not isclose(
            sum(self.weights.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "Risk scenario weights must sum to one.",
            )

        return self


class RiskArtifactPayload(BaseModel):
    """Validate the exported portfolio risk benchmark JSON artifact."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1]
    return_definition: RiskReturnDefinition
    market_splits: RiskMarketSplitsArtifact
    scenarios: tuple[
        RiskScenarioArtifact,
        ...,
    ]

    @model_validator(mode="after")
    def validate_scenarios(
        self,
    ) -> "RiskArtifactPayload":
        if not self.scenarios:
            raise ValueError(
                "Risk artifact must contain at least one scenario.",
            )

        scenario_names = tuple(scenario.name for scenario in self.scenarios)

        if len(scenario_names) != len(set(scenario_names)):
            raise ValueError(
                "Risk artifact scenario names must be unique.",
            )

        training_split = self.market_splits.training
        validation_split = self.market_splits.validation

        expected_tickers = set(training_split.tickers)

        for scenario in self.scenarios:
            if set(scenario.weights) != expected_tickers:
                raise ValueError(
                    "Risk scenario weights must match market tickers.",
                )

            if scenario.training.metrics.observations != training_split.observations:
                raise ValueError(
                    "Training risk observations must match market metadata.",
                )

            if (
                scenario.validation.metrics.observations
                != validation_split.observations
            ):
                raise ValueError(
                    "Validation risk observations must match market metadata.",
                )

            if (
                scenario.training.metrics.periods_per_year
                != training_split.periods_per_year
            ):
                raise ValueError(
                    "Training risk periods_per_year must match market metadata.",
                )

            if (
                scenario.validation.metrics.periods_per_year
                != validation_split.periods_per_year
            ):
                raise ValueError(
                    "Validation risk periods_per_year must match market metadata.",
                )

        return self


class RiskStatusResponse(BaseModel):
    """Describe availability of frozen risk benchmark artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    artifact_available: bool
    validation_correlation_available: bool


class RiskCorrelationRow(BaseModel):
    """Store correlations for one ticker against the market universe."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    ticker: str
    correlations: dict[
        str,
        float,
    ]


class RiskMarketResponse(BaseModel):
    """Expose validation market metadata and the correlation matrix."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evaluation_split: RiskEvaluationSplit
    return_definition: RiskReturnDefinition
    observations: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    date_min: date
    date_max: date
    tickers: tuple[str, ...]
    correlation_matrix: tuple[
        RiskCorrelationRow,
        ...,
    ]


class RiskScenarioSummary(BaseModel):
    """Expose validation risk measures for one portfolio scenario."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    construction: str
    risk: RiskAnalysisArtifact


class RiskScenariosResponse(BaseModel):
    """Compare validation risk across all portfolio scenarios."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    evaluation_split: RiskEvaluationSplit
    return_definition: RiskReturnDefinition
    scenarios: tuple[
        RiskScenarioSummary,
        ...,
    ]


class RiskScenarioDetailResponse(BaseModel):
    """Expose training and validation risk for one scenario."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    name: str
    construction: str
    return_definition: RiskReturnDefinition
    training: RiskAnalysisArtifact
    validation: RiskAnalysisArtifact

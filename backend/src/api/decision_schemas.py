"""HTTP schemas for deterministic portfolio decision support."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.src.decision.models import DecisionProfileType, LiquidityNeed


class DecisionPrioritiesPayload(BaseModel):
    """Four explicit scoring priorities that must sum to one."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    return_priority: float = Field(ge=0.0, le=1.0)
    volatility_priority: float = Field(ge=0.0, le=1.0)
    drawdown_priority: float = Field(ge=0.0, le=1.0)
    diversification_priority: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> DecisionPrioritiesPayload:
        if (
            abs(
                self.return_priority
                + self.volatility_priority
                + self.drawdown_priority
                + self.diversification_priority
                - 1.0
            )
            > 1e-9
        ):
            raise ValueError("Decision priorities must sum to one.")
        return self


class DecisionConstraintsPayload(BaseModel):
    """Hard scenario limits applied before scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    maximum_annualized_volatility: float | None = Field(default=None, gt=0.0)
    maximum_drawdown: float | None = Field(default=None, ge=0.0, le=1.0)
    maximum_concentration_index: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
    )
    minimum_effective_assets: float | None = Field(default=None, ge=1.0)
    minimum_diversification_ratio: float | None = Field(default=None, gt=0.0)
    minimum_cash_reserve: float = Field(default=0.0, ge=0.0, le=1.0)


class DecisionProfilePayload(BaseModel):
    """Preset or complete custom profile accepted by the API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_type: DecisionProfileType
    priorities: DecisionPrioritiesPayload | None = None
    constraints: DecisionConstraintsPayload | None = None
    liquidity_need: LiquidityNeed | None = None
    investment_horizon_years: int | None = Field(default=None, ge=1, le=50)

    @model_validator(mode="after")
    def validate_profile_shape(self) -> DecisionProfilePayload:
        optional_values = (
            self.priorities,
            self.constraints,
            self.liquidity_need,
            self.investment_horizon_years,
        )
        if self.profile_type is DecisionProfileType.CUSTOM:
            if any(value is None for value in optional_values):
                raise ValueError("A custom profile requires every profile field.")
        elif any(value is not None for value in optional_values):
            raise ValueError("Preset profiles cannot be overridden.")
        return self


class DecisionEvaluationRequest(BaseModel):
    """Request a deterministic ranking against frozen portfolio evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    profile: DecisionProfilePayload
    available_liquidity_fraction: float = Field(ge=0.0, le=1.0)


class DecisionProfileResponse(BaseModel):
    """Resolved immutable profile returned to an API consumer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_type: DecisionProfileType
    priorities: dict[str, float]
    constraints: dict[str, float | None]
    liquidity_need: LiquidityNeed
    investment_horizon_years: int


class DecisionProfilesResponse(BaseModel):
    """List supported, fully resolved preset profiles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profiles: tuple[DecisionProfileResponse, ...]


class DecisionWeightResponse(BaseModel):
    """One unchanged weight from the source portfolio scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    ticker: str = Field(min_length=1)
    weight: float = Field(ge=0.0, le=1.0)


class DecisionScenarioResponse(BaseModel):
    """One ranked or rejected portfolio scenario with explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    name: str
    construction: str
    weights: tuple[DecisionWeightResponse, ...]
    admissible: bool
    rank: int | None
    total_score: float | None
    normalized_components: dict[str, float]
    weighted_contributions: dict[str, float]
    constraint_violations: tuple[str, ...]
    tradeoffs: tuple[str, ...]


class DecisionNormalizationResponse(BaseModel):
    """Observed range used for one transparent min-max component."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    metric: str
    minimum: float
    maximum: float
    higher_is_better: bool


class DecisionProvenanceResponse(BaseModel):
    """Validation evidence supporting a recommendation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_schema_version: Literal[1]
    evaluation_split: Literal["validation"]
    return_definition: Literal["adjusted_close_simple_return_on_shared_market_dates"]
    observations: int = Field(gt=0)
    periods_per_year: int = Field(gt=0)
    date_min: date
    date_max: date
    tickers: tuple[str, ...]


class DecisionEvaluationResponse(BaseModel):
    """Complete auditable decision result for one profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deterministic: Literal[True]
    llm_used_for_ranking: Literal[False]
    portfolio_weights_modified: Literal[False]
    provenance: DecisionProvenanceResponse
    profile: DecisionProfileResponse
    available_liquidity_fraction: float = Field(ge=0.0, le=1.0)
    normalization: tuple[DecisionNormalizationResponse, ...]
    recommended_scenario: str | None
    scenarios: tuple[DecisionScenarioResponse, ...]

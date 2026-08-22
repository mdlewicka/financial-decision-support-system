"""Typed and validated inputs for deterministic portfolio decisions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, TypeGuard

Number = int | float


class DecisionProfileType(StrEnum):
    """Supported user profile presets and an explicit custom variant."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    GROWTH = "growth"
    CUSTOM = "custom"


class LiquidityNeed(StrEnum):
    """User-declared need to keep capital outside risky allocations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class DecisionPriorities:
    """Relative importance of the four transparent ranking criteria."""

    return_priority: float
    volatility_priority: float
    drawdown_priority: float
    diversification_priority: float

    _SUM_TOLERANCE = 1e-9

    def __post_init__(self) -> None:
        values = self.as_tuple()

        if any(not _is_finite_number(value) for value in values):
            raise ValueError("Decision priorities must contain finite numbers.")

        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Decision priorities must be in the interval [0, 1].")

        if not math.isclose(
            sum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=self._SUM_TOLERANCE,
        ):
            raise ValueError("Decision priorities must sum to 1.")

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.return_priority,
            self.volatility_priority,
            self.drawdown_priority,
            self.diversification_priority,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "return": self.return_priority,
            "volatility": self.volatility_priority,
            "drawdown": self.drawdown_priority,
            "diversification": self.diversification_priority,
        }


@dataclass(frozen=True, slots=True)
class DecisionConstraints:
    """Hard limits applied before any scenario receives a ranking score."""

    maximum_annualized_volatility: float | None = None
    maximum_drawdown: float | None = None
    maximum_concentration_index: float | None = None
    minimum_effective_assets: float | None = None
    minimum_diversification_ratio: float | None = None
    minimum_cash_reserve: float = 0.0

    def __post_init__(self) -> None:
        _validate_optional_bound(
            self.maximum_annualized_volatility,
            "maximum_annualized_volatility",
            lower=0.0,
            upper=None,
            lower_inclusive=False,
        )
        _validate_optional_bound(
            self.maximum_drawdown,
            "maximum_drawdown",
            lower=0.0,
            upper=1.0,
            lower_inclusive=True,
        )
        _validate_optional_bound(
            self.maximum_concentration_index,
            "maximum_concentration_index",
            lower=0.0,
            upper=1.0,
            lower_inclusive=False,
        )
        _validate_optional_bound(
            self.minimum_effective_assets,
            "minimum_effective_assets",
            lower=1.0,
            upper=None,
            lower_inclusive=True,
        )
        _validate_optional_bound(
            self.minimum_diversification_ratio,
            "minimum_diversification_ratio",
            lower=0.0,
            upper=None,
            lower_inclusive=False,
        )
        _validate_required_bound(
            self.minimum_cash_reserve,
            "minimum_cash_reserve",
            lower=0.0,
            upper=1.0,
        )

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionProfile:
    """Complete, immutable user input consumed by the Decision Engine."""

    profile_type: DecisionProfileType
    priorities: DecisionPriorities
    constraints: DecisionConstraints
    liquidity_need: LiquidityNeed
    investment_horizon_years: int

    def __post_init__(self) -> None:
        if not isinstance(self.profile_type, DecisionProfileType):
            raise TypeError("profile_type must be a DecisionProfileType.")

        if not isinstance(self.priorities, DecisionPriorities):
            raise TypeError("priorities must be a DecisionPriorities instance.")

        if not isinstance(self.constraints, DecisionConstraints):
            raise TypeError("constraints must be a DecisionConstraints instance.")

        if not isinstance(self.liquidity_need, LiquidityNeed):
            raise TypeError("liquidity_need must be a LiquidityNeed.")

        if isinstance(self.investment_horizon_years, bool) or not isinstance(
            self.investment_horizon_years,
            int,
        ):
            raise TypeError("investment_horizon_years must be an integer.")

        if not 1 <= self.investment_horizon_years <= 50:
            raise ValueError("investment_horizon_years must be in [1, 50].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_type": self.profile_type.value,
            "priorities": self.priorities.to_dict(),
            "constraints": self.constraints.to_dict(),
            "liquidity_need": self.liquidity_need.value,
            "investment_horizon_years": self.investment_horizon_years,
        }


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    """Metrics required to compare one frozen portfolio scenario."""

    name: str
    annualized_return: float
    annualized_volatility: float
    maximum_drawdown: float
    concentration_index: float
    effective_number_of_assets: float
    diversification_ratio: float | None

    def __post_init__(self) -> None:
        cleaned_name = self.name.strip()

        if not cleaned_name:
            raise ValueError("Scenario name must not be empty.")

        object.__setattr__(self, "name", cleaned_name)

        for field_name in (
            "annualized_return",
            "annualized_volatility",
            "maximum_drawdown",
            "concentration_index",
            "effective_number_of_assets",
        ):
            if not _is_finite_number(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a finite number.")

        if self.annualized_volatility < 0.0:
            raise ValueError("annualized_volatility must be non-negative.")

        if not 0.0 <= self.maximum_drawdown <= 1.0:
            raise ValueError("maximum_drawdown must be in [0, 1].")

        if not 0.0 < self.concentration_index <= 1.0:
            raise ValueError("concentration_index must be in (0, 1].")

        if self.effective_number_of_assets < 1.0:
            raise ValueError("effective_number_of_assets must be at least 1.")

        if self.diversification_ratio is not None and (
            not _is_finite_number(self.diversification_ratio)
            or self.diversification_ratio <= 0.0
        ):
            raise ValueError(
                "diversification_ratio must be None or a finite positive number."
            )


@dataclass(frozen=True, slots=True)
class MetricNormalization:
    """Observed development range used for transparent min-max scaling."""

    metric: str
    minimum: float
    maximum: float
    higher_is_better: bool


@dataclass(frozen=True, slots=True)
class RankedScenario:
    """One admissible or rejected scenario with an auditable score."""

    name: str
    admissible: bool
    rank: int | None
    total_score: float | None
    normalized_components: dict[str, float]
    weighted_contributions: dict[str, float]
    constraint_violations: tuple[str, ...]
    tradeoffs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionEngineResult:
    """Deterministic ranking plus provenance of every normalization step."""

    profile: DecisionProfile
    available_liquidity_fraction: float
    normalization: tuple[MetricNormalization, ...]
    scenarios: tuple[RankedScenario, ...]
    recommended_scenario: str | None

    @property
    def admissible_scenarios(self) -> tuple[RankedScenario, ...]:
        return tuple(scenario for scenario in self.scenarios if scenario.admissible)

    @property
    def rejected_scenarios(self) -> tuple[RankedScenario, ...]:
        return tuple(scenario for scenario in self.scenarios if not scenario.admissible)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "available_liquidity_fraction": (self.available_liquidity_fraction),
            "normalization": [asdict(item) for item in self.normalization],
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "recommended_scenario": self.recommended_scenario,
        }


def _is_finite_number(value: object) -> TypeGuard[Number]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_required_bound(
    value: object,
    name: str,
    *,
    lower: float,
    upper: float,
) -> None:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number.")

    numeric = float(value)

    if not lower <= numeric <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")


def _validate_optional_bound(
    value: object,
    name: str,
    *,
    lower: float,
    upper: float | None,
    lower_inclusive: bool,
) -> None:
    if value is None:
        return

    if not _is_finite_number(value):
        raise ValueError(f"{name} must be None or a finite number.")

    numeric = float(value)
    lower_valid = numeric >= lower if lower_inclusive else numeric > lower
    upper_valid = upper is None or numeric <= upper

    if not lower_valid or not upper_valid:
        lower_symbol = "[" if lower_inclusive else "("
        upper_text = "infinity" if upper is None else str(upper)
        raise ValueError(f"{name} must be in {lower_symbol}{lower}, {upper_text}].")

"""Frozen profile presets used by the deterministic Decision Engine."""

from __future__ import annotations

from backend.src.decision.models import (
    DecisionConstraints,
    DecisionPriorities,
    DecisionProfile,
    DecisionProfileType,
    LiquidityNeed,
)


def create_decision_profile(
    profile_type: DecisionProfileType,
    *,
    priorities: DecisionPriorities | None = None,
    constraints: DecisionConstraints | None = None,
    liquidity_need: LiquidityNeed | None = None,
    investment_horizon_years: int | None = None,
) -> DecisionProfile:
    """Create a frozen preset or an explicitly complete custom profile."""

    if not isinstance(profile_type, DecisionProfileType):
        raise TypeError("profile_type must be a DecisionProfileType.")

    if profile_type is DecisionProfileType.CUSTOM:
        missing = [
            name
            for name, value in (
                ("priorities", priorities),
                ("constraints", constraints),
                ("liquidity_need", liquidity_need),
                ("investment_horizon_years", investment_horizon_years),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Custom profile requires explicit values for: " + ", ".join(missing)
            )
        return DecisionProfile(
            profile_type=profile_type,
            priorities=priorities,  # type: ignore[arg-type]
            constraints=constraints,  # type: ignore[arg-type]
            liquidity_need=liquidity_need,  # type: ignore[arg-type]
            investment_horizon_years=investment_horizon_years,  # type: ignore[arg-type]
        )

    if any(
        value is not None
        for value in (
            priorities,
            constraints,
            liquidity_need,
            investment_horizon_years,
        )
    ):
        raise ValueError(
            "Preset profiles cannot be partially overridden; use custom instead."
        )

    return _PRESET_BUILDERS[profile_type]()


def _conservative_profile() -> DecisionProfile:
    return DecisionProfile(
        profile_type=DecisionProfileType.CONSERVATIVE,
        priorities=DecisionPriorities(
            return_priority=0.15,
            volatility_priority=0.30,
            drawdown_priority=0.35,
            diversification_priority=0.20,
        ),
        constraints=DecisionConstraints(
            maximum_annualized_volatility=0.14,
            maximum_drawdown=0.22,
            maximum_concentration_index=0.35,
            minimum_effective_assets=3.0,
            minimum_diversification_ratio=1.0,
            minimum_cash_reserve=0.10,
        ),
        liquidity_need=LiquidityNeed.HIGH,
        investment_horizon_years=3,
    )


def _balanced_profile() -> DecisionProfile:
    return DecisionProfile(
        profile_type=DecisionProfileType.BALANCED,
        priorities=DecisionPriorities(
            return_priority=0.30,
            volatility_priority=0.25,
            drawdown_priority=0.25,
            diversification_priority=0.20,
        ),
        constraints=DecisionConstraints(
            maximum_annualized_volatility=0.20,
            maximum_drawdown=0.32,
            maximum_concentration_index=0.40,
            minimum_effective_assets=2.5,
            minimum_diversification_ratio=1.0,
            minimum_cash_reserve=0.05,
        ),
        liquidity_need=LiquidityNeed.MEDIUM,
        investment_horizon_years=5,
    )


def _growth_profile() -> DecisionProfile:
    return DecisionProfile(
        profile_type=DecisionProfileType.GROWTH,
        priorities=DecisionPriorities(
            return_priority=0.50,
            volatility_priority=0.15,
            drawdown_priority=0.15,
            diversification_priority=0.20,
        ),
        constraints=DecisionConstraints(
            maximum_annualized_volatility=0.30,
            maximum_drawdown=0.45,
            maximum_concentration_index=0.50,
            minimum_effective_assets=2.0,
            minimum_diversification_ratio=1.0,
            minimum_cash_reserve=0.0,
        ),
        liquidity_need=LiquidityNeed.LOW,
        investment_horizon_years=10,
    )


_PRESET_BUILDERS = {
    DecisionProfileType.CONSERVATIVE: _conservative_profile,
    DecisionProfileType.BALANCED: _balanced_profile,
    DecisionProfileType.GROWTH: _growth_profile,
}

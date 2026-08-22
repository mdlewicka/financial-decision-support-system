import pytest

from backend.src.decision.models import (
    DecisionConstraints,
    DecisionPriorities,
    DecisionProfileType,
    LiquidityNeed,
)
from backend.src.decision.profiles import create_decision_profile


def test_three_presets_are_deterministic_and_distinct() -> None:
    conservative = create_decision_profile(DecisionProfileType.CONSERVATIVE)
    balanced = create_decision_profile(DecisionProfileType.BALANCED)
    growth = create_decision_profile(DecisionProfileType.GROWTH)

    assert conservative == create_decision_profile(DecisionProfileType.CONSERVATIVE)
    assert conservative.liquidity_need is LiquidityNeed.HIGH
    assert balanced.liquidity_need is LiquidityNeed.MEDIUM
    assert growth.liquidity_need is LiquidityNeed.LOW
    assert (
        conservative.priorities.drawdown_priority > growth.priorities.drawdown_priority
    )
    assert growth.priorities.return_priority > balanced.priorities.return_priority

    conservative_max_drawdown = conservative.constraints.maximum_drawdown
    balanced_max_drawdown = balanced.constraints.maximum_drawdown
    growth_max_drawdown = growth.constraints.maximum_drawdown

    assert conservative_max_drawdown is not None
    assert balanced_max_drawdown is not None
    assert growth_max_drawdown is not None
    assert conservative_max_drawdown < balanced_max_drawdown < growth_max_drawdown


def test_custom_profile_requires_all_explicit_inputs() -> None:
    with pytest.raises(ValueError, match="requires explicit values"):
        create_decision_profile(DecisionProfileType.CUSTOM)

    profile = create_decision_profile(
        DecisionProfileType.CUSTOM,
        priorities=DecisionPriorities(0.4, 0.2, 0.2, 0.2),
        constraints=DecisionConstraints(maximum_drawdown=0.35),
        liquidity_need=LiquidityNeed.MEDIUM,
        investment_horizon_years=6,
    )

    assert profile.profile_type is DecisionProfileType.CUSTOM
    assert profile.investment_horizon_years == 6


def test_presets_cannot_be_silently_overridden() -> None:
    with pytest.raises(ValueError, match="cannot be partially overridden"):
        create_decision_profile(
            DecisionProfileType.BALANCED,
            investment_horizon_years=20,
        )


def test_unknown_profile_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="DecisionProfileType"):
        create_decision_profile("balanced")  # type: ignore[arg-type]

import json

import pytest

from backend.src.decision.models import (
    DecisionConstraints,
    DecisionPriorities,
    DecisionProfile,
    DecisionProfileType,
    LiquidityNeed,
)


def valid_priorities() -> DecisionPriorities:
    return DecisionPriorities(0.30, 0.25, 0.25, 0.20)


def test_priorities_are_explicit_normalized_and_serializable() -> None:
    priorities = valid_priorities()

    assert priorities.as_tuple() == (0.30, 0.25, 0.25, 0.20)
    assert priorities.to_dict() == {
        "return": 0.30,
        "volatility": 0.25,
        "drawdown": 0.25,
        "diversification": 0.20,
    }


@pytest.mark.parametrize(
    "priorities",
    [
        (0.30, 0.25, 0.25, 0.19),
        (-0.10, 0.40, 0.40, 0.30),
        (float("nan"), 0.30, 0.30, 0.40),
        (True, 0.30, 0.30, 0.40),
    ],
)
def test_invalid_priorities_are_rejected(
    priorities: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="priorities"):
        DecisionPriorities(*priorities)


def test_constraints_accept_explicit_limits_and_no_limit_values() -> None:
    unconstrained = DecisionConstraints()
    constrained = DecisionConstraints(
        maximum_annualized_volatility=0.20,
        maximum_drawdown=0.30,
        maximum_concentration_index=0.40,
        minimum_effective_assets=2.5,
        minimum_diversification_ratio=1.1,
        minimum_cash_reserve=0.05,
    )

    assert unconstrained.maximum_drawdown is None
    assert constrained.to_dict()["minimum_cash_reserve"] == 0.05


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"maximum_annualized_volatility": 0.0}, "annualized_volatility"),
        ({"maximum_drawdown": 1.1}, "maximum_drawdown"),
        ({"maximum_concentration_index": 0.0}, "concentration_index"),
        ({"minimum_effective_assets": 0.9}, "effective_assets"),
        ({"minimum_diversification_ratio": 0.0}, "diversification_ratio"),
        ({"minimum_cash_reserve": -0.1}, "cash_reserve"),
        ({"minimum_cash_reserve": float("inf")}, "cash_reserve"),
    ],
)
def test_invalid_constraints_are_rejected(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DecisionConstraints(**kwargs)


def test_complete_profile_is_json_serializable() -> None:
    profile = DecisionProfile(
        profile_type=DecisionProfileType.CUSTOM,
        priorities=valid_priorities(),
        constraints=DecisionConstraints(maximum_drawdown=0.30),
        liquidity_need=LiquidityNeed.MEDIUM,
        investment_horizon_years=7,
    )

    payload = profile.to_dict()

    assert payload["profile_type"] == "custom"
    assert payload["liquidity_need"] == "medium"
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize("horizon", [0, 51])
def test_invalid_investment_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(ValueError, match="investment_horizon_years"):
        DecisionProfile(
            profile_type=DecisionProfileType.CUSTOM,
            priorities=valid_priorities(),
            constraints=DecisionConstraints(),
            liquidity_need=LiquidityNeed.LOW,
            investment_horizon_years=horizon,
        )

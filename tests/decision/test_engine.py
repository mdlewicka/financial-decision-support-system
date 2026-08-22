import json

import pytest

from backend.src.decision.engine import DecisionEngine
from backend.src.decision.models import (
    DecisionConstraints,
    DecisionPriorities,
    DecisionProfileType,
    LiquidityNeed,
    ScenarioMetrics,
)
from backend.src.decision.profiles import create_decision_profile


def scenarios() -> tuple[ScenarioMetrics, ...]:
    return (
        ScenarioMetrics("defensive", 0.06, 0.10, 0.15, 0.25, 4.0, 1.5),
        ScenarioMetrics("balanced", 0.10, 0.16, 0.25, 0.20, 5.0, 1.4),
        ScenarioMetrics("growth", 0.16, 0.25, 0.40, 0.35, 2.8, 1.2),
    )


def custom_profile(
    constraints: DecisionConstraints | None = None,
):
    return create_decision_profile(
        DecisionProfileType.CUSTOM,
        priorities=DecisionPriorities(0.4, 0.2, 0.2, 0.2),
        constraints=constraints or DecisionConstraints(),
        liquidity_need=LiquidityNeed.MEDIUM,
        investment_horizon_years=5,
    )


def test_ranking_is_deterministic_normalized_and_auditable() -> None:
    engine = DecisionEngine()

    first = engine.rank(scenarios(), custom_profile(), available_liquidity_fraction=0.1)
    second = engine.rank(
        tuple(reversed(scenarios())),
        custom_profile(),
        available_liquidity_fraction=0.1,
    )

    assert first == second
    assert first.recommended_scenario is not None
    assert [item.rank for item in first.admissible_scenarios] == [1, 2, 3]

    for item in first.scenarios:
        assert item.total_score is not None
        assert 0.0 <= item.total_score <= 1.0
        assert (
            pytest.approx(sum(item.weighted_contributions.values())) == item.total_score
        )

    assert len(first.normalization) == 4
    json.dumps(first.to_dict(), allow_nan=False)


def test_constraints_reject_scenario_before_scoring() -> None:
    result = DecisionEngine().rank(
        scenarios(),
        custom_profile(
            DecisionConstraints(
                maximum_annualized_volatility=0.20,
                maximum_drawdown=0.30,
                maximum_concentration_index=0.30,
                minimum_effective_assets=3.0,
                minimum_diversification_ratio=1.3,
                minimum_cash_reserve=0.05,
            )
        ),
        available_liquidity_fraction=0.05,
    )

    rejected = result.rejected_scenarios
    assert len(rejected) == 1
    assert rejected[0].name == "growth"
    assert rejected[0].rank is None
    assert rejected[0].total_score is None
    assert set(rejected[0].constraint_violations) == {
        "maximum_annualized_volatility",
        "maximum_drawdown",
        "maximum_concentration_index",
        "minimum_effective_assets",
        "minimum_diversification_ratio",
    }
    return_range = next(
        item for item in result.normalization if item.metric == "return"
    )
    assert return_range.maximum == 0.10


def test_insufficient_external_liquidity_rejects_all_scenarios() -> None:
    result = DecisionEngine().rank(
        scenarios(),
        custom_profile(DecisionConstraints(minimum_cash_reserve=0.20)),
        available_liquidity_fraction=0.10,
    )

    assert result.recommended_scenario is None
    assert not result.admissible_scenarios
    assert result.normalization == ()
    assert all(
        item.constraint_violations == ("minimum_cash_reserve",)
        for item in result.rejected_scenarios
    )


def test_tradeoffs_identify_strongest_and_weakest_components() -> None:
    result = DecisionEngine().rank(
        scenarios(), custom_profile(), available_liquidity_fraction=0.1
    )

    for item in result.admissible_scenarios:
        assert item.tradeoffs[0].startswith("strongest_component:")
        assert item.tradeoffs[1].startswith("weakest_component:")


def test_duplicate_names_and_invalid_liquidity_are_rejected() -> None:
    duplicate = (scenarios()[0], scenarios()[0])
    engine = DecisionEngine()

    with pytest.raises(ValueError, match="unique"):
        engine.rank(duplicate, custom_profile(), available_liquidity_fraction=0.1)
    with pytest.raises(ValueError, match="available_liquidity_fraction"):
        engine.rank(scenarios(), custom_profile(), available_liquidity_fraction=1.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " "}, "name"),
        ({"annualized_volatility": -0.1}, "annualized_volatility"),
        ({"maximum_drawdown": 1.1}, "maximum_drawdown"),
        ({"concentration_index": 0.0}, "concentration_index"),
        ({"effective_number_of_assets": 0.5}, "effective_number_of_assets"),
        ({"diversification_ratio": 0.0}, "diversification_ratio"),
    ],
)
def test_invalid_scenario_metrics_are_rejected(
    kwargs: dict[str, str | float],
    message: str,
) -> None:
    values = {
        "name": "scenario",
        "annualized_return": 0.1,
        "annualized_volatility": 0.2,
        "maximum_drawdown": 0.3,
        "concentration_index": 0.25,
        "effective_number_of_assets": 4.0,
        "diversification_ratio": 1.3,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        ScenarioMetrics(**values)  # type: ignore[arg-type]

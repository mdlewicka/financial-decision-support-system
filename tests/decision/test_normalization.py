import pytest

from backend.src.decision.normalization import (
    NEUTRAL_CONSTANT_SCORE,
    build_normalization,
    normalize_value,
)


def test_benefit_and_cost_normalization_are_explicit() -> None:
    benefit = build_normalization("return", (0.05, 0.10, 0.15), higher_is_better=True)
    cost = build_normalization("volatility", (0.10, 0.20, 0.30), higher_is_better=False)

    assert normalize_value(0.05, benefit) == 0.0
    assert normalize_value(0.15, benefit) == 1.0
    assert normalize_value(0.10, cost) == 1.0
    assert normalize_value(0.30, cost) == 0.0


def test_constant_metric_receives_neutral_score() -> None:
    normalization = build_normalization(
        "drawdown", (0.20, 0.20), higher_is_better=False
    )

    assert normalize_value(0.20, normalization) == NEUTRAL_CONSTANT_SCORE


def test_invalid_normalization_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_normalization("return", (), higher_is_better=True)
    with pytest.raises(ValueError, match="finite"):
        build_normalization("return", (0.1, float("nan")), higher_is_better=True)

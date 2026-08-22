"""Tests for the validation-only portfolio artifact adapter."""

from datetime import date

import pytest

from backend.src.api.risk_schemas import RiskArtifactPayload
from backend.src.decision.portfolio_adapter import (
    PortfolioArtifactDecisionAdapter,
)


def test_adapter_uses_validation_metrics_and_preserves_provenance(
    portfolio_artifact: RiskArtifactPayload,
) -> None:
    result = PortfolioArtifactDecisionAdapter().adapt(portfolio_artifact)

    assert [scenario.name for scenario in result.scenarios] == [
        "balanced",
        "defensive",
        "growth",
    ]
    assert result.scenarios[0].annualized_return == pytest.approx(0.10)
    assert result.scenarios[0].annualized_volatility == pytest.approx(0.15)
    assert result.scenarios[0].annualized_return < 1.0
    assert result.provenance.evaluation_split == "validation"
    assert result.provenance.observations == 8
    assert result.provenance.date_min == date(2020, 2, 1)
    assert result.provenance.date_max == date(2020, 2, 12)


def test_adapter_preserves_fixed_weights_without_mutable_aliases(
    portfolio_artifact: RiskArtifactPayload,
) -> None:
    result = PortfolioArtifactDecisionAdapter().adapt(portfolio_artifact)
    context = result.context_for("defensive")

    assert context.construction == "fixed_defensive"
    assert dict(context.weights) == {
        "AGG": 0.25,
        "GLD": 0.25,
        "QQQ": 0.25,
        "SPY": 0.25,
    }
    assert isinstance(context.weights, tuple)


def test_adapter_rejects_unvalidated_inputs() -> None:
    with pytest.raises(TypeError, match="RiskArtifactPayload"):
        PortfolioArtifactDecisionAdapter().adapt({})  # type: ignore[arg-type]

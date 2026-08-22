"""Shared validated portfolio artifacts for Decision Engine tests."""

from __future__ import annotations

import pytest

from backend.src.api.risk_schemas import RiskArtifactPayload


def _analysis(
    *,
    observations: int,
    annualized_return: float,
    annualized_volatility: float,
    maximum_drawdown: float,
    concentration_index: float,
    effective_assets: float,
    diversification_ratio: float,
) -> dict[str, object]:
    return {
        "metrics": {
            "observations": observations,
            "periods_per_year": 252,
            "confidence_level": 0.95,
            "annual_risk_free_rate": 0.02,
            "mean_return": annualized_return / 252,
            "annualized_mean_return": annualized_return,
            "volatility": annualized_volatility / (252**0.5),
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": 0.5,
            "sortino_ratio": 0.7,
            "value_at_risk": 0.01,
            "conditional_value_at_risk": 0.02,
            "maximum_drawdown": maximum_drawdown,
        },
        "concentration_index": concentration_index,
        "effective_number_of_assets": effective_assets,
        "diversification_ratio": diversification_ratio,
    }


@pytest.fixture
def portfolio_artifact() -> RiskArtifactPayload:
    tickers = ("AGG", "GLD", "QQQ", "SPY")
    specifications = (
        ("defensive", 0.05, 0.10, 0.12, 0.30, 3.33, 1.20),
        ("balanced", 0.10, 0.15, 0.20, 0.25, 4.00, 1.30),
        ("growth", 0.15, 0.25, 0.35, 0.40, 2.50, 1.10),
    )
    scenarios = []
    for (
        name,
        annual_return,
        volatility,
        drawdown,
        hhi,
        effective,
        ratio,
    ) in specifications:
        scenarios.append(
            {
                "name": name,
                "construction": f"fixed_{name}",
                "weights": {ticker: 0.25 for ticker in tickers},
                "training": _analysis(
                    observations=20,
                    annualized_return=annual_return + 10.0,
                    annualized_volatility=volatility + 1.0,
                    maximum_drawdown=drawdown,
                    concentration_index=hhi,
                    effective_assets=effective,
                    diversification_ratio=ratio,
                ),
                "validation": _analysis(
                    observations=8,
                    annualized_return=annual_return,
                    annualized_volatility=volatility,
                    maximum_drawdown=drawdown,
                    concentration_index=hhi,
                    effective_assets=effective,
                    diversification_ratio=ratio,
                ),
            }
        )
    return RiskArtifactPayload.model_validate(
        {
            "schema_version": 1,
            "return_definition": (
                "adjusted_close_simple_return_on_shared_market_dates"
            ),
            "market_splits": {
                "training": {
                    "observations": 20,
                    "periods_per_year": 252,
                    "date_min": "2020-01-01",
                    "date_max": "2020-01-31",
                    "tickers": tickers,
                },
                "validation": {
                    "observations": 8,
                    "periods_per_year": 252,
                    "date_min": "2020-02-01",
                    "date_max": "2020-02-12",
                    "tickers": tickers,
                },
            },
            "scenarios": scenarios,
        }
    )

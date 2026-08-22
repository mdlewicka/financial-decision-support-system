from backend.src.risk.risk_metrics import RiskMetrics


def test_risk_metrics_stores_reproducibility_metadata_and_values() -> None:
    metrics = RiskMetrics(
        observations=100,
        periods_per_year=252,
        confidence_level=0.95,
        annual_risk_free_rate=0.02,
        mean_return=0.001,
        annualized_mean_return=0.252,
        volatility=0.01,
        annualized_volatility=0.158745,
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        value_at_risk=0.02,
        conditional_value_at_risk=0.03,
        maximum_drawdown=0.15,
    )

    assert metrics.observations == 100
    assert metrics.periods_per_year == 252
    assert metrics.confidence_level == 0.95
    assert metrics.annual_risk_free_rate == 0.02
    assert metrics.mean_return == 0.001
    assert metrics.annualized_mean_return == 0.252
    assert metrics.volatility == 0.01
    assert metrics.annualized_volatility == 0.158745
    assert metrics.sharpe_ratio == 1.2
    assert metrics.sortino_ratio == 1.8
    assert metrics.value_at_risk == 0.02
    assert metrics.conditional_value_at_risk == 0.03
    assert metrics.maximum_drawdown == 0.15

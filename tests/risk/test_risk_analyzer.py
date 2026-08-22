import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.risk.risk_analyzer import (
    RiskAnalysisConfig,
    RiskAnalyzer,
)
from backend.src.risk.risk_metrics import RiskMetrics


def create_returns(values: list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2025-01-01", periods=len(values), freq="D"),
        dtype=float,
    )


def test_analyze_calculates_complete_risk_metrics() -> None:
    returns = create_returns(
        [
            0.01,
            -0.02,
            0.03,
            -0.01,
            0.02,
        ]
    )
    config = RiskAnalysisConfig(
        confidence_level=0.80,
        periods_per_year=12,
        minimum_observations=5,
    )

    metrics = RiskAnalyzer(config=config).analyze(returns)

    values = returns.to_numpy(dtype=float)
    expected_mean = float(np.mean(values))
    expected_volatility = float(np.std(values, ddof=1))
    downside_returns = np.minimum(values, 0.0)
    expected_downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))
    expected_threshold = float(np.quantile(values, 0.20, method="linear"))
    expected_tail_mean = float(np.mean(values[values <= expected_threshold]))

    wealth_index = np.concatenate(
        (
            np.array([1.0]),
            np.cumprod(1.0 + values),
        )
    )
    expected_drawdown = float(
        np.max(1.0 - wealth_index / np.maximum.accumulate(wealth_index))
    )

    assert isinstance(metrics, RiskMetrics)
    assert metrics.observations == 5
    assert metrics.periods_per_year == 12
    assert metrics.confidence_level == 0.80
    assert metrics.annual_risk_free_rate == 0.0
    assert metrics.mean_return == pytest.approx(expected_mean)
    assert metrics.annualized_mean_return == pytest.approx(expected_mean * 12)
    assert metrics.volatility == pytest.approx(expected_volatility)
    assert metrics.annualized_volatility == pytest.approx(
        expected_volatility * math.sqrt(12)
    )
    assert metrics.sharpe_ratio == pytest.approx(
        expected_mean / expected_volatility * math.sqrt(12)
    )
    assert metrics.sortino_ratio == pytest.approx(
        expected_mean / expected_downside_deviation * math.sqrt(12)
    )
    assert metrics.value_at_risk == pytest.approx(max(0.0, -expected_threshold))
    assert metrics.conditional_value_at_risk == pytest.approx(
        max(0.0, -expected_tail_mean)
    )
    assert metrics.maximum_drawdown == pytest.approx(expected_drawdown)


def test_maximum_drawdown_includes_initial_wealth() -> None:
    returns = create_returns([-0.20, 0.10, 0.05])
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(minimum_observations=3),
    )

    metrics = analyzer.analyze(returns)

    assert metrics.maximum_drawdown == pytest.approx(0.20)


def test_total_loss_produces_full_drawdown() -> None:
    returns = create_returns([0.05, -1.0, 0.20])
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(minimum_observations=3),
    )

    metrics = analyzer.analyze(returns)

    assert metrics.maximum_drawdown == pytest.approx(1.0)


def test_sortino_uses_downside_deviation_over_all_observations() -> None:
    returns = create_returns([0.04, -0.02, 0.01, -0.01])
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(
            periods_per_year=4,
            minimum_observations=4,
        ),
    )

    metrics = analyzer.analyze(returns)

    values = returns.to_numpy(dtype=float)
    downside_deviation = float(np.sqrt(np.mean(np.square(np.minimum(values, 0.0)))))
    expected = float(np.mean(values)) / downside_deviation * math.sqrt(4)

    assert metrics.sortino_ratio == pytest.approx(expected)


def test_risk_free_rate_is_converted_to_periodic_compound_rate() -> None:
    returns = create_returns([0.01, 0.02, 0.03, 0.04])
    config = RiskAnalysisConfig(
        annual_risk_free_rate=0.10,
        periods_per_year=4,
        minimum_observations=4,
    )

    metrics = RiskAnalyzer(config=config).analyze(returns)

    periodic_rate = math.expm1(math.log1p(0.10) / 4)
    excess_mean = float(np.mean(returns.to_numpy(dtype=float) - periodic_rate))
    volatility = float(np.std(returns.to_numpy(dtype=float), ddof=1))

    assert metrics.sharpe_ratio == pytest.approx(
        excess_mean / volatility * math.sqrt(4)
    )


def test_constant_returns_produce_undefined_ratios_instead_of_infinity() -> None:
    returns = create_returns([0.0] * 30)

    metrics = RiskAnalyzer().analyze(returns)

    assert metrics.sharpe_ratio is None
    assert metrics.sortino_ratio is None


def test_tail_risk_is_reported_as_non_negative_loss() -> None:
    returns = create_returns([0.01, 0.02, 0.03, 0.04])
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(
            confidence_level=0.75,
            minimum_observations=4,
        ),
    )

    metrics = analyzer.analyze(returns)

    assert metrics.value_at_risk == 0.0
    assert metrics.conditional_value_at_risk == 0.0


def test_bitcoin_calendar_can_use_365_periods_per_year() -> None:
    returns = create_returns([0.01, -0.02, 0.03, -0.01])
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(
            periods_per_year=365,
            minimum_observations=4,
        ),
    )

    metrics = analyzer.analyze(returns)

    expected = returns.std() * math.sqrt(365)

    assert metrics.periods_per_year == 365
    assert metrics.annualized_volatility == pytest.approx(expected)


def test_analyze_does_not_modify_input_series() -> None:
    returns = create_returns([0.01, -0.02, 0.03])
    original = returns.copy(deep=True)
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(minimum_observations=3),
    )

    analyzer.analyze(returns)

    pdt.assert_series_equal(returns, original)


@pytest.mark.parametrize(
    ("returns", "error_type", "message"),
    [
        (
            [0.01, 0.02, 0.03],
            TypeError,
            "returns must be a pandas Series",
        ),
        (
            pd.Series(["0.01", "0.02", "0.03"]),
            TypeError,
            "returns must contain numeric values",
        ),
        (
            pd.Series([True, False, True]),
            TypeError,
            "returns must contain numeric values",
        ),
        (
            pd.Series([0.01, 0.02]),
            ValueError,
            "returns must contain at least 3 observations",
        ),
        (
            pd.Series([0.01, np.nan, 0.03]),
            ValueError,
            "returns must contain only finite values",
        ),
        (
            pd.Series([0.01, np.inf, 0.03]),
            ValueError,
            "returns must contain only finite values",
        ),
        (
            pd.Series([0.01, -1.01, 0.03]),
            ValueError,
            "returns cannot be lower than -1.0",
        ),
        (
            pd.Series([0.01, 0.02, 0.03], index=[0, 0, 1]),
            ValueError,
            "returns index must not contain duplicates",
        ),
        (
            pd.Series([0.01, 0.02, 0.03], index=[2, 0, 1]),
            ValueError,
            "returns index must be sorted in ascending order",
        ),
    ],
)
def test_analyze_rejects_invalid_returns(
    returns: object,
    error_type: type[Exception],
    message: str,
) -> None:
    analyzer = RiskAnalyzer(
        config=RiskAnalysisConfig(minimum_observations=3),
    )

    with pytest.raises(error_type, match=message):
        analyzer.analyze(returns)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("create_config", "message"),
    [
        (
            lambda: RiskAnalysisConfig(confidence_level=0.0),
            "confidence_level must be between 0 and 1",
        ),
        (
            lambda: RiskAnalysisConfig(confidence_level=1.0),
            "confidence_level must be between 0 and 1",
        ),
        (
            lambda: RiskAnalysisConfig(confidence_level=float("nan")),
            "confidence_level must be finite",
        ),
        (
            lambda: RiskAnalysisConfig(annual_risk_free_rate=-1.0),
            "annual_risk_free_rate must be greater than -1",
        ),
        (
            lambda: RiskAnalysisConfig(periods_per_year=0),
            "periods_per_year must be a positive integer",
        ),
        (
            lambda: RiskAnalysisConfig(minimum_observations=1),
            "minimum_observations must be an integer of at least 2",
        ),
    ],
)
def test_config_rejects_invalid_values(
    create_config: Callable[[], RiskAnalysisConfig],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_config()

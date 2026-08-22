from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.portfolio.markowitz_optimizer import (
    MarkowitzOptimizationConfig,
    MarkowitzOptimizer,
)
from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_metrics import (
    PortfolioAnalysisResult,
    PortfolioObjective,
    PortfolioOptimizationResult,
)
from backend.src.risk.risk_analyzer import RiskAnalysisConfig


def create_returns() -> pd.DataFrame:
    cycles = 12
    return pd.DataFrame(
        {
            "SPY": [0.020, -0.015, 0.018, -0.010, 0.012, -0.005] * cycles,
            "AGG": [0.002, 0.001, 0.003, 0.000, 0.002, 0.001] * cycles,
            "GLD": [-0.005, 0.010, -0.003, 0.008, 0.001, 0.004] * cycles,
            "QQQ": [0.025, -0.020, 0.022, -0.012, 0.018, -0.008] * cycles,
        },
        index=pd.date_range("2024-01-01", periods=6 * cycles, freq="D"),
        dtype=float,
    )


def create_analysis_config() -> PortfolioAnalysisConfig:
    return PortfolioAnalysisConfig(
        risk_config=RiskAnalysisConfig(
            annual_risk_free_rate=0.02,
            periods_per_year=252,
            minimum_observations=30,
        )
    )


def create_optimizer(
    maximum_weight: float = 0.60,
) -> MarkowitzOptimizer:
    return MarkowitzOptimizer(
        config=MarkowitzOptimizationConfig(
            analysis_config=create_analysis_config(),
            maximum_weight=maximum_weight,
        )
    )


def analyze_equal_weight_portfolio() -> PortfolioAnalysisResult:
    returns = create_returns()
    weights = pd.Series(0.25, index=returns.columns, dtype=float)
    return PortfolioAnalyzer(config=create_analysis_config()).analyze(
        aligned_returns=returns,
        weights=weights,
    )


@pytest.mark.parametrize(
    "objective",
    [
        PortfolioObjective.MINIMUM_VOLATILITY,
        PortfolioObjective.MAXIMUM_SHARPE,
    ],
)
def test_optimize_returns_valid_long_only_allocation(
    objective: PortfolioObjective,
) -> None:
    result = create_optimizer().optimize(create_returns(), objective)

    assert isinstance(result, PortfolioOptimizationResult)
    assert result.objective is objective
    assert result.analysis.weights.sum() == pytest.approx(1.0)
    assert result.analysis.weights.ge(0.0).all()
    assert result.analysis.weights.le(0.60 + 1e-8).all()
    assert result.maximum_weight == 0.60
    assert result.solver_iterations > 0
    assert result.solver_message


def test_minimum_volatility_improves_on_equal_weight_baseline() -> None:
    baseline = analyze_equal_weight_portfolio()
    result = create_optimizer().optimize(
        create_returns(),
        PortfolioObjective.MINIMUM_VOLATILITY,
    )

    baseline_volatility = baseline.risk_metrics.annualized_volatility
    assert result.analysis.risk_metrics.annualized_volatility <= (
        baseline_volatility + 1e-8
    )


def test_maximum_sharpe_improves_on_equal_weight_baseline() -> None:
    baseline = analyze_equal_weight_portfolio()
    result = create_optimizer().optimize(
        create_returns(),
        PortfolioObjective.MAXIMUM_SHARPE,
    )

    baseline_sharpe = baseline.risk_metrics.sharpe_ratio
    optimized_sharpe = result.analysis.risk_metrics.sharpe_ratio

    assert baseline_sharpe is not None
    assert optimized_sharpe is not None
    assert optimized_sharpe >= baseline_sharpe - 1e-8


def test_optimize_is_deterministic() -> None:
    optimizer = create_optimizer()

    first = optimizer.optimize(
        create_returns(),
        PortfolioObjective.MAXIMUM_SHARPE,
    )
    second = optimizer.optimize(
        create_returns(),
        PortfolioObjective.MAXIMUM_SHARPE,
    )

    pdt.assert_series_equal(first.analysis.weights, second.analysis.weights)


def test_optimize_does_not_modify_input_returns() -> None:
    returns = create_returns()
    original = returns.copy(deep=True)

    create_optimizer().optimize(
        returns,
        PortfolioObjective.MINIMUM_VOLATILITY,
    )

    pdt.assert_frame_equal(returns, original)


def test_optimize_rejects_infeasible_maximum_weight() -> None:
    optimizer = create_optimizer(maximum_weight=0.20)

    with pytest.raises(ValueError, match="maximum_weight is infeasible"):
        optimizer.optimize(
            create_returns(),
            PortfolioObjective.MINIMUM_VOLATILITY,
        )


def test_maximum_sharpe_rejects_undefined_ratio() -> None:
    returns = pd.DataFrame(
        0.0,
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
        columns=["AGG", "GLD"],
    )
    optimizer = create_optimizer(maximum_weight=1.0)

    with pytest.raises(RuntimeError, match="undefined Sharpe ratio"):
        optimizer.optimize(returns, PortfolioObjective.MAXIMUM_SHARPE)


def test_minimum_volatility_accepts_zero_volatility_solution() -> None:
    returns = pd.DataFrame(
        0.0,
        index=pd.date_range("2024-01-01", periods=30, freq="D"),
        columns=["AGG", "GLD"],
    )
    optimizer = create_optimizer(maximum_weight=1.0)

    result = optimizer.optimize(
        returns,
        PortfolioObjective.MINIMUM_VOLATILITY,
    )

    assert result.analysis.risk_metrics.annualized_volatility == 0.0


def test_optimize_rejects_unknown_objective() -> None:
    with pytest.raises(TypeError, match="objective must be a PortfolioObjective"):
        create_optimizer().optimize(
            create_returns(),
            cast(PortfolioObjective, "unknown"),
        )


def test_optimize_propagates_return_validation() -> None:
    returns = create_returns().assign(SPY=np.inf)

    with pytest.raises(ValueError, match="aligned_returns must contain only finite"):
        create_optimizer().optimize(
            returns,
            PortfolioObjective.MINIMUM_VOLATILITY,
        )


def test_optimize_reports_solver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = SimpleNamespace(
        success=False,
        message="iteration limit reached",
    )
    monkeypatch.setattr(
        "backend.src.portfolio.markowitz_optimizer.minimize",
        lambda *args, **kwargs: failure,
    )

    with pytest.raises(RuntimeError, match="iteration limit reached"):
        create_optimizer().optimize(
            create_returns(),
            PortfolioObjective.MINIMUM_VOLATILITY,
        )


@pytest.mark.parametrize(
    ("create_config", "error_type", "message"),
    [
        (
            lambda: MarkowitzOptimizationConfig(
                analysis_config=cast(PortfolioAnalysisConfig, "invalid")
            ),
            TypeError,
            "analysis_config must be a PortfolioAnalysisConfig",
        ),
        (
            lambda: MarkowitzOptimizationConfig(maximum_weight=0.0),
            ValueError,
            "maximum_weight must be between 0 and 1",
        ),
        (
            lambda: MarkowitzOptimizationConfig(maximum_weight=1.01),
            ValueError,
            "maximum_weight must be between 0 and 1",
        ),
        (
            lambda: MarkowitzOptimizationConfig(solver_tolerance=0.0),
            ValueError,
            "solver_tolerance must be greater than zero",
        ),
        (
            lambda: MarkowitzOptimizationConfig(maximum_iterations=0),
            ValueError,
            "maximum_iterations must be a positive integer",
        ),
    ],
)
def test_config_rejects_invalid_values(
    create_config: Callable[[], MarkowitzOptimizationConfig],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        create_config()

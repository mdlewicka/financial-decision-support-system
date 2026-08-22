import math
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_metrics import PortfolioAnalysisResult
from backend.src.risk.risk_analyzer import RiskAnalysisConfig


def create_returns() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SPY": [0.01, -0.02, 0.03, 0.01, -0.01],
            "AGG": [0.002, 0.001, -0.001, 0.003, 0.000],
            "GLD": [-0.01, 0.02, 0.01, -0.005, 0.015],
        },
        index=pd.date_range(
            "2025-01-01",
            periods=5,
            freq="D",
        ),
        dtype=float,
    )


def create_weights() -> pd.Series:
    return pd.Series(
        {
            "SPY": 0.50,
            "AGG": 0.30,
            "GLD": 0.20,
        },
        dtype=float,
    )


def create_analyzer(
    weight_sum_tolerance: float = 1e-8,
) -> PortfolioAnalyzer:
    return PortfolioAnalyzer(
        config=PortfolioAnalysisConfig(
            risk_config=RiskAnalysisConfig(
                confidence_level=0.80,
                periods_per_year=12,
                minimum_observations=5,
            ),
            weight_sum_tolerance=weight_sum_tolerance,
        )
    )


def test_analyze_calculates_complete_portfolio_result() -> None:
    returns = create_returns()
    weights = create_weights()

    result = create_analyzer().analyze(
        returns,
        weights,
    )

    ordered_returns = returns.reindex(
        columns=[
            "AGG",
            "GLD",
            "SPY",
        ]
    )
    ordered_weights = weights.reindex(
        [
            "AGG",
            "GLD",
            "SPY",
        ]
    )

    expected_portfolio_returns = ordered_returns.mul(
        ordered_weights,
        axis="columns",
    ).sum(axis="columns")

    expected_portfolio_returns = expected_portfolio_returns.rename("portfolio_return")

    assert isinstance(
        result,
        PortfolioAnalysisResult,
    )

    pdt.assert_series_equal(
        result.weights,
        ordered_weights.rename("weight"),
    )

    pdt.assert_series_equal(
        result.portfolio_returns,
        expected_portfolio_returns,
    )

    assert result.risk_metrics.observations == 5

    assert result.risk_metrics.periods_per_year == 12

    assert result.risk_metrics.mean_return == pytest.approx(
        expected_portfolio_returns.mean()
    )

    assert result.risk_metrics.annualized_mean_return == pytest.approx(
        expected_portfolio_returns.mean() * 12
    )

    assert result.risk_metrics.annualized_volatility == pytest.approx(
        expected_portfolio_returns.std(ddof=1) * math.sqrt(12)
    )


def test_return_contributions_sum_to_portfolio_expected_return() -> None:
    result = create_analyzer().analyze(
        create_returns(),
        create_weights(),
    )

    assert result.annualized_return_contributions.sum() == pytest.approx(
        result.risk_metrics.annualized_mean_return
    )


def test_volatility_contributions_sum_to_portfolio_volatility() -> None:
    result = create_analyzer().analyze(
        create_returns(),
        create_weights(),
    )

    assert result.annualized_volatility_contributions.sum() == pytest.approx(
        result.risk_metrics.annualized_volatility
    )


def test_analyze_calculates_concentration_and_diversification() -> None:
    returns = create_returns()
    weights = create_weights()

    result = create_analyzer().analyze(
        returns,
        weights,
    )

    ordered_weights = result.weights

    weight_values = np.asarray(
        ordered_weights,
        dtype=np.float64,
    )

    expected_concentration = float(np.sum(np.square(weight_values)))

    individual_volatilities = returns.reindex(columns=ordered_weights.index).std(
        ddof=1
    ) * math.sqrt(12)

    individual_volatility_values = np.asarray(
        individual_volatilities,
        dtype=np.float64,
    )

    weighted_individual_volatility = float(
        np.dot(
            weight_values,
            individual_volatility_values,
        )
    )

    expected_diversification = (
        weighted_individual_volatility / result.risk_metrics.annualized_volatility
    )

    assert result.concentration_index == pytest.approx(expected_concentration)

    assert result.effective_number_of_assets == pytest.approx(
        1.0 / expected_concentration
    )

    assert result.diversification_ratio == pytest.approx(expected_diversification)

    assert result.diversification_ratio is not None

    assert result.diversification_ratio >= 1.0


def test_analyze_aligns_weights_to_sorted_tickers() -> None:
    returns = create_returns().reindex(
        columns=[
            "GLD",
            "SPY",
            "AGG",
        ]
    )

    weights = create_weights().reindex(
        [
            "SPY",
            "AGG",
            "GLD",
        ]
    )

    result = create_analyzer().analyze(
        returns,
        weights,
    )

    assert list(result.weights.index) == [
        "AGG",
        "GLD",
        "SPY",
    ]

    assert list(result.annualized_return_contributions.index) == [
        "AGG",
        "GLD",
        "SPY",
    ]


def test_analyze_normalizes_weights_within_tolerance() -> None:
    weights = pd.Series(
        {
            "SPY": 0.50,
            "AGG": 0.30,
            "GLD": 0.2000001,
        },
        dtype=float,
    )

    analyzer = create_analyzer(weight_sum_tolerance=1e-6)

    result = analyzer.analyze(
        create_returns(),
        weights,
    )

    assert result.weights.sum() == pytest.approx(1.0)


def test_analyze_allows_zero_weight_as_explicit_scenario() -> None:
    weights = pd.Series(
        {
            "SPY": 1.0,
            "AGG": 0.0,
            "GLD": 0.0,
        },
        dtype=float,
    )

    result = create_analyzer().analyze(
        create_returns(),
        weights,
    )

    pdt.assert_series_equal(
        result.portfolio_returns,
        create_returns()["SPY"].rename("portfolio_return"),
    )

    assert result.concentration_index == pytest.approx(1.0)

    assert result.effective_number_of_assets == pytest.approx(1.0)


def test_zero_volatility_portfolio_has_defined_contributions() -> None:
    returns = pd.DataFrame(
        {
            "AGG": [0.0] * 5,
            "GLD": [0.0] * 5,
        },
        index=pd.date_range(
            "2025-01-01",
            periods=5,
            freq="D",
        ),
        dtype=float,
    )

    weights = pd.Series(
        {
            "AGG": 0.5,
            "GLD": 0.5,
        },
        dtype=float,
    )

    result = create_analyzer().analyze(
        returns,
        weights,
    )

    assert result.risk_metrics.annualized_volatility == 0.0

    assert result.risk_metrics.sharpe_ratio is None

    assert result.annualized_volatility_contributions.eq(0.0).all()

    assert result.diversification_ratio is None


def test_analyze_does_not_modify_inputs() -> None:
    returns = create_returns()
    weights = create_weights()

    original_returns = returns.copy(deep=True)
    original_weights = weights.copy(deep=True)

    create_analyzer().analyze(
        returns,
        weights,
    )

    pdt.assert_frame_equal(
        returns,
        original_returns,
    )

    pdt.assert_series_equal(
        weights,
        original_weights,
    )


@pytest.mark.parametrize(
    (
        "transform",
        "error_type",
        "message",
    ),
    [
        (
            lambda data: data.assign(SPY="0.01"),
            TypeError,
            ("aligned_returns must contain only numeric values"),
        ),
        (
            lambda data: data.assign(SPY=np.inf),
            ValueError,
            ("aligned_returns must contain only finite values"),
        ),
        (
            lambda data: data.assign(SPY=-1.01),
            ValueError,
            ("aligned_returns cannot contain returns lower than -1.0"),
        ),
        (
            lambda data: data.iloc[::-1],
            ValueError,
            ("aligned_returns index must be sorted"),
        ),
    ],
)
def test_analyze_rejects_invalid_return_matrices(
    transform: Callable[
        [pd.DataFrame],
        pd.DataFrame,
    ],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        error_type,
        match=message,
    ):
        create_analyzer().analyze(
            transform(create_returns()),
            create_weights(),
        )


def test_analyze_rejects_non_dataframe_returns() -> None:
    with pytest.raises(
        TypeError,
        match=("aligned_returns must be a pandas DataFrame"),
    ):
        create_analyzer().analyze(
            [],  # type: ignore[arg-type]
            create_weights(),
        )


def test_analyze_rejects_too_few_assets() -> None:
    returns = create_returns().loc[
        :,
        ["SPY"],
    ]

    weights = pd.Series(
        {
            "SPY": 1.0,
        },
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="at least two assets",
    ):
        create_analyzer().analyze(
            returns,
            weights,
        )


def test_analyze_rejects_too_few_observations() -> None:
    with pytest.raises(
        ValueError,
        match="at least 5 observations",
    ):
        create_analyzer().analyze(
            create_returns().iloc[:4],
            create_weights(),
        )


def test_analyze_rejects_duplicate_return_columns() -> None:
    returns = create_returns()

    returns.columns = [
        "SPY",
        "SPY",
        "GLD",
    ]

    with pytest.raises(
        ValueError,
        match=("columns must not contain duplicates"),
    ):
        create_analyzer().analyze(
            returns,
            create_weights(),
        )


def test_analyze_rejects_duplicate_return_index() -> None:
    returns = create_returns()

    returns.index = pd.Index(
        [
            0,
            0,
            1,
            2,
            3,
        ]
    )

    with pytest.raises(
        ValueError,
        match=("index must not contain duplicates"),
    ):
        create_analyzer().analyze(
            returns,
            create_weights(),
        )


@pytest.mark.parametrize(
    (
        "weights",
        "error_type",
        "message",
    ),
    [
        (
            [0.5, 0.3, 0.2],
            TypeError,
            ("weights must be a pandas Series"),
        ),
        (
            pd.Series(
                [
                    0.5,
                    0.3,
                    0.2,
                ],
                index=[
                    "SPY",
                    "SPY",
                    "GLD",
                ],
                dtype=float,
            ),
            ValueError,
            ("weights index must not contain duplicate tickers"),
        ),
        (
            pd.Series(
                {
                    "SPY": True,
                    "AGG": False,
                    "GLD": True,
                }
            ),
            TypeError,
            ("weights must contain numeric values"),
        ),
        (
            pd.Series(
                {
                    "SPY": 0.5,
                    "AGG": 0.5,
                }
            ),
            ValueError,
            ("weights tickers must exactly match"),
        ),
        (
            pd.Series(
                {
                    "SPY": 0.5,
                    "AGG": 0.3,
                    "GLD": 0.1,
                    "QQQ": 0.1,
                }
            ),
            ValueError,
            ("weights tickers must exactly match"),
        ),
        (
            pd.Series(
                {
                    "SPY": 0.5,
                    "AGG": -0.1,
                    "GLD": 0.6,
                }
            ),
            ValueError,
            ("weights must be non-negative"),
        ),
        (
            pd.Series(
                {
                    "SPY": 0.5,
                    "AGG": 0.3,
                    "GLD": np.nan,
                }
            ),
            ValueError,
            ("weights must contain only finite values"),
        ),
        (
            pd.Series(
                {
                    "SPY": 0.5,
                    "AGG": 0.3,
                    "GLD": 0.3,
                }
            ),
            ValueError,
            ("weights must sum to 1"),
        ),
    ],
)
def test_analyze_rejects_invalid_weights(
    weights: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        error_type,
        match=message,
    ):
        create_analyzer().analyze(
            create_returns(),
            cast(
                pd.Series,
                weights,
            ),
        )


@pytest.mark.parametrize(
    (
        "create_config",
        "error_type",
        "message",
    ),
    [
        (
            lambda: PortfolioAnalysisConfig(
                risk_config=cast(
                    RiskAnalysisConfig,
                    "invalid",
                )
            ),
            TypeError,
            ("risk_config must be a RiskAnalysisConfig"),
        ),
        (
            lambda: PortfolioAnalysisConfig(weight_sum_tolerance=0.0),
            ValueError,
            ("weight_sum_tolerance must be greater than zero"),
        ),
        (
            lambda: PortfolioAnalysisConfig(weight_sum_tolerance=float("nan")),
            ValueError,
            ("weight_sum_tolerance must be finite"),
        ),
    ],
)
def test_config_rejects_invalid_values(
    create_config: Callable[
        [],
        PortfolioAnalysisConfig,
    ],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(
        error_type,
        match=message,
    ):
        create_config()

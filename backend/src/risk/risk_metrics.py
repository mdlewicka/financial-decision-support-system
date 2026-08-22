from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class RiskMetrics:
    """Risk metrics calculated for one ordered return series."""

    observations: int
    periods_per_year: int
    confidence_level: float
    annual_risk_free_rate: float
    mean_return: float
    annualized_mean_return: float
    volatility: float
    annualized_volatility: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    value_at_risk: float
    conditional_value_at_risk: float
    maximum_drawdown: float


@dataclass(frozen=True, slots=True)
class MultiAssetRiskResult:
    """Risk matrices calculated from returns on shared market dates."""

    observations: int
    periods_per_year: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    tickers: tuple[str, ...]
    aligned_returns: pd.DataFrame
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame
    annualized_covariance_matrix: pd.DataFrame

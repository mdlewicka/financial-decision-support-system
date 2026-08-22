from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from backend.src.risk.risk_metrics import RiskMetrics


@dataclass(frozen=True, slots=True)
class PortfolioAnalysisResult:
    """Complete analysis of one explicit portfolio allocation."""

    weights: pd.Series
    portfolio_returns: pd.Series
    risk_metrics: RiskMetrics
    annualized_return_contributions: pd.Series
    annualized_volatility_contributions: pd.Series
    concentration_index: float
    effective_number_of_assets: float
    diversification_ratio: float | None


class PortfolioObjective(StrEnum):
    """Supported, intentionally compact Markowitz objectives."""

    MINIMUM_VOLATILITY = "minimum_volatility"
    MAXIMUM_SHARPE = "maximum_sharpe"


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationResult:
    """Validated allocation returned by the numerical optimizer."""

    objective: PortfolioObjective
    analysis: PortfolioAnalysisResult
    maximum_weight: float
    solver_iterations: int
    solver_message: str


@dataclass(frozen=True, slots=True)
class PortfolioScenarioEvaluation:
    """Training and validation analysis for one fixed allocation."""

    name: str
    construction: str
    training: PortfolioAnalysisResult
    validation: PortfolioAnalysisResult

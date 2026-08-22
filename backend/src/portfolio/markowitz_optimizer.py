from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.src.portfolio.portfolio_analyzer import (
    PortfolioAnalysisConfig,
    PortfolioAnalyzer,
)
from backend.src.portfolio.portfolio_metrics import (
    PortfolioObjective,
    PortfolioOptimizationResult,
)


@dataclass(frozen=True, slots=True)
class MarkowitzOptimizationConfig:
    """Configuration for constrained, long-only Markowitz optimization."""

    analysis_config: PortfolioAnalysisConfig = field(
        default_factory=PortfolioAnalysisConfig
    )
    maximum_weight: float = 0.40
    solver_tolerance: float = 1e-9
    maximum_iterations: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.analysis_config, PortfolioAnalysisConfig):
            raise TypeError(
                "analysis_config must be a PortfolioAnalysisConfig instance."
            )

        if isinstance(self.maximum_weight, bool) or not math.isfinite(
            self.maximum_weight
        ):
            raise ValueError("maximum_weight must be finite.")
        if not 0.0 < self.maximum_weight <= 1.0:
            raise ValueError("maximum_weight must be between 0 and 1.")

        if isinstance(self.solver_tolerance, bool) or not math.isfinite(
            self.solver_tolerance
        ):
            raise ValueError("solver_tolerance must be finite.")
        if self.solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be greater than zero.")

        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, int)
            or self.maximum_iterations <= 0
        ):
            raise ValueError("maximum_iterations must be a positive integer.")


class MarkowitzOptimizer:
    """Optimize a validated return matrix using two transparent objectives."""

    _NUMERICAL_TOLERANCE = 1e-8
    _ZERO_TOLERANCE = 1e-15
    _UNDEFINED_SHARPE_PENALTY = 1e12

    def __init__(
        self,
        config: MarkowitzOptimizationConfig | None = None,
    ) -> None:
        self.config = config or MarkowitzOptimizationConfig()
        self._portfolio_analyzer = PortfolioAnalyzer(config=self.config.analysis_config)

    def optimize(
        self,
        aligned_returns: pd.DataFrame,
        objective: PortfolioObjective,
    ) -> PortfolioOptimizationResult:
        """Optimize weights and return their complete portfolio analysis."""

        if not isinstance(objective, PortfolioObjective):
            raise TypeError("objective must be a PortfolioObjective value.")

        validated_returns, initial_weights = self._prepare_returns(aligned_returns)
        expected_returns = (
            validated_returns.mean()
            * self.config.analysis_config.risk_config.periods_per_year
        )
        covariance_matrix = (
            validated_returns.cov()
            * self.config.analysis_config.risk_config.periods_per_year
        )
        self._validate_covariance_matrix(covariance_matrix)

        bounds = tuple(
            (0.0, self.config.maximum_weight) for _ in range(validated_returns.shape[1])
        )
        optimization_result = minimize(
            fun=self._objective_function,
            x0=initial_weights.to_numpy(dtype=float),
            args=(
                objective,
                expected_returns.to_numpy(dtype=float),
                covariance_matrix.to_numpy(dtype=float),
            ),
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": self._weight_constraint},
            options={
                "ftol": self.config.solver_tolerance,
                "maxiter": self.config.maximum_iterations,
            },
        )

        if not optimization_result.success:
            raise RuntimeError(
                f"Portfolio optimization failed: {optimization_result.message}"
            )

        optimized_weights = self._validate_optimized_weights(
            raw_weights=np.asarray(optimization_result.x, dtype=float),
            tickers=tuple(cast(str, ticker) for ticker in validated_returns.columns),
        )
        analysis = self._portfolio_analyzer.analyze(
            aligned_returns=validated_returns,
            weights=optimized_weights,
        )

        if (
            objective is PortfolioObjective.MAXIMUM_SHARPE
            and analysis.risk_metrics.sharpe_ratio is None
        ):
            raise RuntimeError(
                "Maximum Sharpe optimization produced an undefined Sharpe ratio."
            )

        return PortfolioOptimizationResult(
            objective=objective,
            analysis=analysis,
            maximum_weight=self.config.maximum_weight,
            solver_iterations=int(optimization_result.nit),
            solver_message=str(optimization_result.message),
        )

    def _prepare_returns(
        self,
        aligned_returns: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if not isinstance(aligned_returns, pd.DataFrame):
            raise TypeError("aligned_returns must be a pandas DataFrame.")

        number_of_assets = aligned_returns.shape[1]
        if number_of_assets == 0:
            raise ValueError("aligned_returns must contain assets.")

        if (
            self.config.maximum_weight * number_of_assets
            < 1.0 - self._NUMERICAL_TOLERANCE
        ):
            raise ValueError(
                "maximum_weight is infeasible for the number of available assets."
            )

        equal_weights = pd.Series(
            1.0 / number_of_assets,
            index=aligned_returns.columns,
            dtype=float,
        )
        baseline_analysis = self._portfolio_analyzer.analyze(
            aligned_returns=aligned_returns,
            weights=equal_weights,
        )
        tickers = tuple(baseline_analysis.weights.index)
        validated_returns = (
            aligned_returns.reindex(columns=tickers).astype(float).copy()
        )

        return validated_returns, baseline_analysis.weights

    def _objective_function(
        self,
        weights: np.ndarray,
        objective: PortfolioObjective,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        volatility = self._portfolio_volatility(
            weights=weights,
            covariance_matrix=covariance_matrix,
        )

        if objective is PortfolioObjective.MINIMUM_VOLATILITY:
            return volatility

        if volatility <= self._ZERO_TOLERANCE:
            return self._UNDEFINED_SHARPE_PENALTY

        portfolio_return = float(weights @ expected_returns)
        risk_free_rate = self.config.analysis_config.risk_config.annual_risk_free_rate
        sharpe_ratio = (portfolio_return - risk_free_rate) / volatility

        return (
            -sharpe_ratio
            if math.isfinite(sharpe_ratio)
            else (self._UNDEFINED_SHARPE_PENALTY)
        )

    @staticmethod
    def _weight_constraint(
        weights: np.ndarray,
    ) -> float:
        return float(weights.sum() - 1.0)

    @classmethod
    def _portfolio_volatility(
        cls,
        weights: np.ndarray,
        covariance_matrix: np.ndarray,
    ) -> float:
        variance = float(weights @ covariance_matrix @ weights)
        if variance < -cls._NUMERICAL_TOLERANCE:
            return math.inf

        return math.sqrt(max(0.0, variance))

    @classmethod
    def _validate_covariance_matrix(
        cls,
        covariance_matrix: pd.DataFrame,
    ) -> None:
        covariance_values = covariance_matrix.to_numpy(dtype=float)
        if not np.isfinite(covariance_values).all():
            raise ValueError("annualized covariance matrix must be finite.")

        if not np.allclose(
            covariance_values,
            covariance_values.T,
            atol=cls._NUMERICAL_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("annualized covariance matrix must be symmetric.")

        minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
        if minimum_eigenvalue < -cls._NUMERICAL_TOLERANCE:
            raise ValueError(
                "annualized covariance matrix must be positive semidefinite."
            )

    def _validate_optimized_weights(
        self,
        raw_weights: np.ndarray,
        tickers: tuple[str, ...],
    ) -> pd.Series:
        if not np.isfinite(raw_weights).all():
            raise RuntimeError("Portfolio optimization returned non-finite weights.")

        if (raw_weights < -self._NUMERICAL_TOLERANCE).any() or (
            raw_weights > self.config.maximum_weight + self._NUMERICAL_TOLERANCE
        ).any():
            raise RuntimeError(
                "Portfolio optimization returned weights outside configured bounds."
            )

        weight_sum = float(raw_weights.sum())
        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=self._NUMERICAL_TOLERANCE,
        ):
            raise RuntimeError(
                "Portfolio optimization returned weights that do not sum to 1."
            )

        clipped_weights = np.clip(
            raw_weights,
            0.0,
            self.config.maximum_weight,
        )
        normalized_weights = clipped_weights / clipped_weights.sum()
        if (
            normalized_weights > self.config.maximum_weight + self._NUMERICAL_TOLERANCE
        ).any():
            raise RuntimeError(
                "Normalized portfolio weights exceed the configured maximum."
            )

        return pd.Series(
            normalized_weights,
            index=tickers,
            name="weight",
            dtype=float,
        )

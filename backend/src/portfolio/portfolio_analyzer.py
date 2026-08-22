from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from backend.src.portfolio.portfolio_metrics import PortfolioAnalysisResult
from backend.src.risk.risk_analyzer import RiskAnalysisConfig, RiskAnalyzer


@dataclass(frozen=True, slots=True)
class PortfolioAnalysisConfig:
    """Configuration for evaluating one explicit portfolio allocation."""

    risk_config: RiskAnalysisConfig = field(
        default_factory=RiskAnalysisConfig,
    )
    weight_sum_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(
            self.risk_config,
            RiskAnalysisConfig,
        ):
            raise TypeError("risk_config must be a RiskAnalysisConfig instance.")

        if isinstance(
            self.weight_sum_tolerance,
            bool,
        ) or not math.isfinite(
            self.weight_sum_tolerance,
        ):
            raise ValueError("weight_sum_tolerance must be finite.")

        if self.weight_sum_tolerance <= 0.0:
            raise ValueError("weight_sum_tolerance must be greater than zero.")


class PortfolioAnalyzer:
    """Evaluate return and risk for weights chosen by the user."""

    _ZERO_TOLERANCE = 1e-15

    def __init__(
        self,
        config: PortfolioAnalysisConfig | None = None,
    ) -> None:
        self.config = config or PortfolioAnalysisConfig()
        self._risk_analyzer = RiskAnalyzer(
            config=self.config.risk_config,
        )

    def analyze(
        self,
        aligned_returns: pd.DataFrame,
        weights: pd.Series,
    ) -> PortfolioAnalysisResult:
        """Analyze one long-only allocation on a complete return matrix."""

        validated_returns = self._validate_returns(
            aligned_returns,
        )
        validated_weights = self._validate_weights(
            weights=weights,
            tickers=tuple(validated_returns.columns),
        )

        portfolio_returns = validated_returns.mul(
            validated_weights,
            axis="columns",
        ).sum(axis="columns")
        portfolio_returns = portfolio_returns.astype(float).rename("portfolio_return")

        risk_metrics = self._risk_analyzer.analyze(
            portfolio_returns,
        )
        annualized_return_contributions = self._calculate_return_contributions(
            aligned_returns=validated_returns,
            weights=validated_weights,
        )
        annualized_volatility_contributions = self._calculate_volatility_contributions(
            aligned_returns=validated_returns,
            weights=validated_weights,
            portfolio_volatility=(risk_metrics.annualized_volatility),
        )
        concentration_index = float(np.square(validated_weights).sum())
        effective_number_of_assets = 1.0 / concentration_index
        diversification_ratio = self._calculate_diversification_ratio(
            aligned_returns=validated_returns,
            weights=validated_weights,
            portfolio_volatility=(risk_metrics.annualized_volatility),
        )

        return PortfolioAnalysisResult(
            weights=validated_weights,
            portfolio_returns=portfolio_returns,
            risk_metrics=risk_metrics,
            annualized_return_contributions=(annualized_return_contributions),
            annualized_volatility_contributions=(annualized_volatility_contributions),
            concentration_index=concentration_index,
            effective_number_of_assets=(effective_number_of_assets),
            diversification_ratio=diversification_ratio,
        )

    def _validate_returns(
        self,
        aligned_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(
            aligned_returns,
            pd.DataFrame,
        ):
            raise TypeError("aligned_returns must be a pandas DataFrame.")

        if aligned_returns.shape[1] < 2:
            raise ValueError("aligned_returns must contain at least two assets.")

        minimum_observations = self.config.risk_config.minimum_observations
        if len(aligned_returns) < minimum_observations:
            raise ValueError(
                "aligned_returns must contain at least "
                f"{minimum_observations} observations."
            )

        if aligned_returns.columns.has_duplicates:
            raise ValueError("aligned_returns columns must not contain duplicates.")

        valid_columns = all(
            isinstance(column, str) and bool(column.strip())
            for column in aligned_returns.columns
        )
        if not valid_columns:
            raise ValueError(
                "aligned_returns columns must contain unique non-empty ticker names."
            )

        if aligned_returns.index.has_duplicates:
            raise ValueError("aligned_returns index must not contain duplicates.")

        if not aligned_returns.index.is_monotonic_increasing:
            raise ValueError("aligned_returns index must be sorted in ascending order.")

        for column in aligned_returns.columns:
            values = cast(
                pd.Series,
                aligned_returns[column],
            )
            if not pd.api.types.is_numeric_dtype(
                values.dtype,
            ) or pd.api.types.is_bool_dtype(
                values.dtype,
            ):
                raise TypeError("aligned_returns must contain only numeric values.")

        tickers = tuple(sorted(cast(str, value) for value in aligned_returns.columns))
        validated_returns = (
            aligned_returns.reindex(
                columns=tickers,
            )
            .astype(float)
            .copy()
        )
        return_values = validated_returns.to_numpy(
            dtype=float,
        )

        if not np.isfinite(return_values).all():
            raise ValueError("aligned_returns must contain only finite values.")

        if (return_values < -1.0).any():
            raise ValueError("aligned_returns cannot contain returns lower than -1.0.")

        return validated_returns

    def _validate_weights(
        self,
        weights: pd.Series,
        tickers: tuple[str, ...],
    ) -> pd.Series:
        if not isinstance(weights, pd.Series):
            raise TypeError("weights must be a pandas Series.")

        if weights.index.has_duplicates:
            raise ValueError("weights index must not contain duplicate tickers.")

        valid_index = all(
            isinstance(ticker, str) and bool(ticker.strip()) for ticker in weights.index
        )
        if not valid_index:
            raise ValueError("weights index must contain non-empty ticker names.")

        if not pd.api.types.is_numeric_dtype(
            weights.dtype,
        ) or pd.api.types.is_bool_dtype(
            weights.dtype,
        ):
            raise TypeError("weights must contain numeric values.")

        weight_tickers = {cast(str, ticker) for ticker in weights.index}
        expected_tickers = set(tickers)

        if weight_tickers != expected_tickers:
            missing_tickers = sorted(
                expected_tickers.difference(
                    weight_tickers,
                )
            )
            unexpected_tickers = sorted(
                weight_tickers.difference(
                    expected_tickers,
                )
            )
            details: list[str] = []

            if missing_tickers:
                details.append("missing: " + ", ".join(missing_tickers))

            if unexpected_tickers:
                details.append("unexpected: " + ", ".join(unexpected_tickers))

            raise ValueError(
                "weights tickers must exactly match "
                "aligned_returns columns (" + "; ".join(details) + ")."
            )

        validated_weights = weights.reindex(tickers).astype(float).copy()
        weight_values = validated_weights.to_numpy(
            dtype=float,
        )

        if not np.isfinite(weight_values).all():
            raise ValueError("weights must contain only finite values.")

        if (weight_values < 0.0).any():
            raise ValueError("weights must be non-negative.")

        weight_sum = float(weight_values.sum())
        if not math.isclose(
            weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=self.config.weight_sum_tolerance,
        ):
            raise ValueError("weights must sum to 1 within the configured tolerance.")

        normalized_weights = validated_weights / weight_sum

        return normalized_weights.rename(
            "weight",
        )

    def _calculate_return_contributions(
        self,
        aligned_returns: pd.DataFrame,
        weights: pd.Series,
    ) -> pd.Series:
        annualized_asset_returns = (
            aligned_returns.mean() * self.config.risk_config.periods_per_year
        )
        contributions = weights * annualized_asset_returns

        return contributions.astype(float).rename(
            "annualized_return_contribution",
        )

    def _calculate_volatility_contributions(
        self,
        aligned_returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_volatility: float,
    ) -> pd.Series:
        if portfolio_volatility <= self._ZERO_TOLERANCE:
            return pd.Series(
                0.0,
                index=weights.index,
                name=("annualized_volatility_contribution"),
                dtype=float,
            )

        annualized_covariance = (
            aligned_returns.cov() * self.config.risk_config.periods_per_year
        )
        covariance_values = annualized_covariance.to_numpy(
            dtype=float,
        )
        weight_values = weights.to_numpy(
            dtype=float,
        )
        marginal_risk = covariance_values @ weight_values / portfolio_volatility
        component_risk = weight_values * marginal_risk

        return pd.Series(
            component_risk,
            index=weights.index,
            name=("annualized_volatility_contribution"),
            dtype=float,
        )

    def _calculate_diversification_ratio(
        self,
        aligned_returns: pd.DataFrame,
        weights: pd.Series,
        portfolio_volatility: float,
    ) -> float | None:
        if portfolio_volatility <= self._ZERO_TOLERANCE:
            return None

        individual_volatility_values = aligned_returns.std(ddof=1).to_numpy(
            dtype=float,
        ) * math.sqrt(self.config.risk_config.periods_per_year)
        weight_values = weights.to_numpy(
            dtype=float,
        )
        weighted_individual_volatility = float(
            np.dot(
                weight_values,
                individual_volatility_values,
            )
        )
        ratio = weighted_individual_volatility / portfolio_volatility

        return ratio if math.isfinite(ratio) else None

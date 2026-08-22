from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from backend.src.risk.risk_metrics import MultiAssetRiskResult, RiskMetrics


@dataclass(frozen=True, slots=True)
class RiskAnalysisConfig:
    """Configuration shared by all metrics for one return series."""

    confidence_level: float = 0.95
    annual_risk_free_rate: float = 0.0
    periods_per_year: int = 252
    minimum_observations: int = 30

    def __post_init__(self) -> None:
        if isinstance(self.confidence_level, bool) or not math.isfinite(
            self.confidence_level
        ):
            raise ValueError("confidence_level must be finite.")

        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between 0 and 1.")

        if isinstance(self.annual_risk_free_rate, bool) or not math.isfinite(
            self.annual_risk_free_rate
        ):
            raise ValueError("annual_risk_free_rate must be finite.")

        if self.annual_risk_free_rate <= -1.0:
            raise ValueError("annual_risk_free_rate must be greater than -1.")

        if (
            isinstance(self.periods_per_year, bool)
            or not isinstance(self.periods_per_year, int)
            or self.periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer.")

        if (
            isinstance(self.minimum_observations, bool)
            or not isinstance(self.minimum_observations, int)
            or self.minimum_observations < 2
        ):
            raise ValueError("minimum_observations must be an integer of at least 2.")


class RiskAnalyzer:
    """Calculate validated historical risk metrics for one return series."""

    _ZERO_TOLERANCE = 1e-15

    def __init__(
        self,
        config: RiskAnalysisConfig | None = None,
    ) -> None:
        self.config = config or RiskAnalysisConfig()

    def analyze(
        self,
        returns: pd.Series,
    ) -> RiskMetrics:
        """Calculate risk metrics without modifying the input series."""

        validated_returns = self._validate_returns(returns)
        return_values = validated_returns.to_numpy(dtype=float)

        mean_return = float(np.mean(return_values))
        volatility = float(np.std(return_values, ddof=1))
        annualized_mean_return = mean_return * self.config.periods_per_year
        annualized_volatility = volatility * math.sqrt(self.config.periods_per_year)

        periodic_risk_free_rate = math.expm1(
            math.log1p(self.config.annual_risk_free_rate) / self.config.periods_per_year
        )
        excess_returns = return_values - periodic_risk_free_rate

        sharpe_ratio = self._calculate_sharpe_ratio(
            excess_returns=excess_returns,
            volatility=volatility,
        )
        sortino_ratio = self._calculate_sortino_ratio(
            excess_returns=excess_returns,
        )
        value_at_risk, conditional_value_at_risk = self._calculate_tail_risk(
            return_values=return_values,
        )
        maximum_drawdown = self._calculate_maximum_drawdown(
            return_values=return_values,
        )

        return RiskMetrics(
            observations=len(return_values),
            periods_per_year=self.config.periods_per_year,
            confidence_level=self.config.confidence_level,
            annual_risk_free_rate=self.config.annual_risk_free_rate,
            mean_return=mean_return,
            annualized_mean_return=annualized_mean_return,
            volatility=volatility,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            value_at_risk=value_at_risk,
            conditional_value_at_risk=conditional_value_at_risk,
            maximum_drawdown=maximum_drawdown,
        )

    def _validate_returns(
        self,
        returns: pd.Series,
    ) -> pd.Series:
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pandas Series.")

        if not pd.api.types.is_numeric_dtype(
            returns.dtype
        ) or pd.api.types.is_bool_dtype(returns.dtype):
            raise TypeError("returns must contain numeric values.")

        if len(returns) < self.config.minimum_observations:
            raise ValueError(
                "returns must contain at least "
                f"{self.config.minimum_observations} observations."
            )

        if returns.index.has_duplicates:
            raise ValueError("returns index must not contain duplicates.")

        if not returns.index.is_monotonic_increasing:
            raise ValueError("returns index must be sorted in ascending order.")

        validated_returns = returns.astype(float).copy()
        return_values = validated_returns.to_numpy(dtype=float)

        if not np.isfinite(return_values).all():
            raise ValueError("returns must contain only finite values.")

        if (return_values < -1.0).any():
            raise ValueError("returns cannot be lower than -1.0.")

        return validated_returns

    def _calculate_sharpe_ratio(
        self,
        excess_returns: np.ndarray,
        volatility: float,
    ) -> float | None:
        if volatility <= self._ZERO_TOLERANCE:
            return None

        ratio = (
            float(np.mean(excess_returns))
            / volatility
            * math.sqrt(self.config.periods_per_year)
        )

        return self._finite_ratio_or_none(ratio)

    def _calculate_sortino_ratio(
        self,
        excess_returns: np.ndarray,
    ) -> float | None:
        downside_returns = np.minimum(excess_returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

        if downside_deviation <= self._ZERO_TOLERANCE:
            return None

        ratio = (
            float(np.mean(excess_returns))
            / downside_deviation
            * math.sqrt(self.config.periods_per_year)
        )

        return self._finite_ratio_or_none(ratio)

    def _calculate_tail_risk(
        self,
        return_values: np.ndarray,
    ) -> tuple[float, float]:
        tail_probability = 1.0 - self.config.confidence_level
        return_threshold = float(
            np.quantile(
                return_values,
                tail_probability,
                method="linear",
            )
        )
        tail_returns = return_values[return_values <= return_threshold]

        value_at_risk = max(0.0, -return_threshold)
        conditional_value_at_risk = max(
            0.0,
            -float(np.mean(tail_returns)),
        )

        return value_at_risk, conditional_value_at_risk

    @staticmethod
    def _calculate_maximum_drawdown(
        return_values: np.ndarray,
    ) -> float:
        wealth_index = np.empty(len(return_values) + 1, dtype=float)
        wealth_index[0] = 1.0
        wealth_index[1:] = np.cumprod(1.0 + return_values)
        running_maximum = np.maximum.accumulate(wealth_index)
        drawdowns = 1.0 - wealth_index / running_maximum

        return float(np.max(drawdowns))

    @staticmethod
    def _finite_ratio_or_none(
        ratio: float,
    ) -> float | None:
        return ratio if math.isfinite(ratio) else None


@dataclass(frozen=True, slots=True)
class MultiAssetRiskAnalysisConfig:
    """Configuration for shared-date multi-asset risk matrices."""

    periods_per_year: int = 252
    minimum_common_observations: int = 30
    date_column: str = "date"
    ticker_column: str = "ticker"
    price_column: str = "adjusted_close"

    def __post_init__(self) -> None:
        if (
            isinstance(self.periods_per_year, bool)
            or not isinstance(self.periods_per_year, int)
            or self.periods_per_year <= 0
        ):
            raise ValueError("periods_per_year must be a positive integer.")

        if (
            isinstance(self.minimum_common_observations, bool)
            or not isinstance(self.minimum_common_observations, int)
            or self.minimum_common_observations < 2
        ):
            raise ValueError(
                "minimum_common_observations must be an integer of at least 2."
            )

        for field_name in ("date_column", "ticker_column", "price_column"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")


class MultiAssetRiskAnalyzer:
    """Build complete return, correlation and covariance matrices."""

    def __init__(
        self,
        config: MultiAssetRiskAnalysisConfig | None = None,
    ) -> None:
        self.config = config or MultiAssetRiskAnalysisConfig()

    def analyze(
        self,
        market_data: pd.DataFrame,
    ) -> MultiAssetRiskResult:
        """Calculate matrices from adjusted prices observed on shared dates."""

        validated_data = self._validate_market_data(market_data)
        ticker_values = cast(
            pd.Series,
            validated_data[self.config.ticker_column],
        )
        tickers = tuple(sorted(ticker_values.unique()))

        common_prices = (
            validated_data.pivot(
                index=self.config.date_column,
                columns=self.config.ticker_column,
                values=self.config.price_column,
            )
            .reindex(columns=tickers)
            .dropna(axis="index", how="any")
            .sort_index()
        )

        required_price_observations = self.config.minimum_common_observations + 1
        if len(common_prices) < required_price_observations:
            raise ValueError(
                "market_data must contain at least "
                f"{required_price_observations} shared price dates to calculate "
                f"{self.config.minimum_common_observations} common returns."
            )

        aligned_returns = common_prices.pct_change(fill_method=None).iloc[1:].copy()
        self._validate_aligned_returns(aligned_returns)

        correlation_matrix = aligned_returns.corr()
        covariance_matrix = aligned_returns.cov()
        annualized_covariance_matrix = covariance_matrix * self.config.periods_per_year

        self._validate_finite_matrix(
            matrix=correlation_matrix,
            matrix_name="correlation matrix",
        )
        self._validate_finite_matrix(
            matrix=covariance_matrix,
            matrix_name="covariance matrix",
        )

        return MultiAssetRiskResult(
            observations=len(aligned_returns),
            periods_per_year=self.config.periods_per_year,
            date_min=pd.Timestamp(aligned_returns.index.min()),
            date_max=pd.Timestamp(aligned_returns.index.max()),
            tickers=tickers,
            aligned_returns=aligned_returns,
            correlation_matrix=correlation_matrix,
            covariance_matrix=covariance_matrix,
            annualized_covariance_matrix=annualized_covariance_matrix,
        )

    def _validate_market_data(
        self,
        market_data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(market_data, pd.DataFrame):
            raise TypeError("market_data must be a pandas DataFrame.")

        required_columns = {
            self.config.date_column,
            self.config.ticker_column,
            self.config.price_column,
        }
        missing_columns = sorted(required_columns.difference(market_data.columns))
        if missing_columns:
            raise ValueError(
                "market_data is missing required columns: "
                + ", ".join(missing_columns)
                + "."
            )

        validated_data = market_data.loc[:, sorted(required_columns)].copy()
        date_values = cast(pd.Series, validated_data[self.config.date_column])
        ticker_values = cast(pd.Series, validated_data[self.config.ticker_column])
        price_values = cast(pd.Series, validated_data[self.config.price_column])

        if not pd.api.types.is_datetime64_any_dtype(date_values.dtype):
            raise TypeError(f"{self.config.date_column} must contain datetime values.")
        if date_values.isna().any():
            raise ValueError(
                f"{self.config.date_column} must not contain missing values."
            )

        valid_tickers = ticker_values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        )
        if not bool(valid_tickers.all()):
            raise ValueError(
                f"{self.config.ticker_column} must contain non-empty strings."
            )

        if not pd.api.types.is_numeric_dtype(
            price_values.dtype
        ) or pd.api.types.is_bool_dtype(price_values.dtype):
            raise TypeError(f"{self.config.price_column} must contain numeric values.")

        numeric_prices = price_values.to_numpy(dtype=float)
        if not np.isfinite(numeric_prices).all():
            raise ValueError(
                f"{self.config.price_column} must contain only finite values."
            )
        if (numeric_prices <= 0.0).any():
            raise ValueError(
                f"{self.config.price_column} must contain positive values."
            )

        duplicate_mask = validated_data.duplicated(
            subset=[self.config.ticker_column, self.config.date_column],
            keep=False,
        )
        if duplicate_mask.any():
            raise ValueError("market_data must not contain duplicate ticker-date rows.")

        if ticker_values.nunique() < 2:
            raise ValueError("market_data must contain at least two tickers.")

        return validated_data

    @staticmethod
    def _validate_aligned_returns(
        aligned_returns: pd.DataFrame,
    ) -> None:
        return_values = aligned_returns.to_numpy(dtype=float)
        if not np.isfinite(return_values).all():
            raise ValueError("aligned returns must contain only finite values.")

    @staticmethod
    def _validate_finite_matrix(
        matrix: pd.DataFrame,
        matrix_name: str,
    ) -> None:
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(
                f"{matrix_name} contains undefined values; "
                "each asset must have variable returns."
            )

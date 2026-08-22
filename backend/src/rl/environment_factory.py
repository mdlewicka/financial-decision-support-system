from dataclasses import dataclass

import pandas as pd

from backend.src.risk.risk_analyzer import (
    MultiAssetRiskAnalysisConfig,
    MultiAssetRiskAnalyzer,
)
from backend.src.rl.portfolio_allocation_env import (
    PortfolioAllocationEnv,
    PortfolioAllocationEnvConfig,
)


@dataclass(frozen=True, slots=True)
class PortfolioAllocationEnvironmentBundle:
    training_environment: PortfolioAllocationEnv
    validation_environment: PortfolioAllocationEnv
    tickers: tuple[str, ...]
    training_returns: pd.DataFrame
    validation_returns: pd.DataFrame
    validation_context_returns: pd.DataFrame


class PortfolioAllocationEnvironmentFactory:
    def __init__(
        self,
        environment_config: PortfolioAllocationEnvConfig | None = None,
        minimum_common_observations: int = 30,
        periods_per_year: int = 252,
    ) -> None:
        self.environment_config = environment_config or PortfolioAllocationEnvConfig()
        self._risk_analyzer = MultiAssetRiskAnalyzer(
            config=MultiAssetRiskAnalysisConfig(
                periods_per_year=periods_per_year,
                minimum_common_observations=(minimum_common_observations),
            )
        )

    def create(
        self,
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
    ) -> PortfolioAllocationEnvironmentBundle:
        training_data = self._copy_and_validate_dataset(
            training_dataset,
            dataset_name="training_dataset",
        )
        validation_data = self._copy_and_validate_dataset(
            validation_dataset,
            dataset_name="validation_dataset",
        )

        training_dates = self._extract_dates(
            training_data,
            dataset_name="training_dataset",
        )
        validation_dates = self._extract_dates(
            validation_data,
            dataset_name="validation_dataset",
        )
        training_tickers = self._extract_tickers(
            training_data,
            dataset_name="training_dataset",
        )
        validation_tickers = self._extract_tickers(
            validation_data,
            dataset_name="validation_dataset",
        )

        self._validate_split_boundaries(
            training_dates=training_dates,
            validation_dates=validation_dates,
        )
        self._validate_ticker_sets(
            training_tickers=training_tickers,
            validation_tickers=validation_tickers,
        )

        training_analysis = self._risk_analyzer.analyze(training_data)
        combined_analysis = self._risk_analyzer.analyze(
            pd.concat(
                [
                    training_data,
                    validation_data,
                ],
                ignore_index=True,
            )
        )

        if training_analysis.tickers != combined_analysis.tickers:
            raise ValueError("Training and combined market tickers must match.")

        training_returns = training_analysis.aligned_returns.copy(deep=True)
        validation_start_date = pd.Timestamp(validation_dates.to_numpy().min())
        validation_returns = combined_analysis.aligned_returns.loc[
            combined_analysis.aligned_returns.index >= validation_start_date
        ].copy(deep=True)

        if validation_returns.empty:
            raise ValueError(
                "Validation data must produce at least one shared-date return."
            )

        validation_context_returns = training_returns.tail(
            self.environment_config.lookback_window
        ).copy(deep=True)

        if len(validation_context_returns) < self.environment_config.lookback_window:
            raise ValueError(
                "Training data does not contain enough shared "
                "returns for the validation lookback context."
            )

        if validation_context_returns.index.max() >= validation_returns.index.min():
            raise ValueError(
                "Validation returns must begin after the training context."
            )

        validation_environment_returns = pd.concat(
            [
                validation_context_returns,
                validation_returns,
            ],
            axis="index",
        )

        training_environment = PortfolioAllocationEnv(
            returns=training_returns,
            config=self.environment_config,
        )
        validation_environment = PortfolioAllocationEnv(
            returns=validation_environment_returns,
            config=self.environment_config,
        )

        return PortfolioAllocationEnvironmentBundle(
            training_environment=training_environment,
            validation_environment=validation_environment,
            tickers=training_analysis.tickers,
            training_returns=training_returns.copy(deep=True),
            validation_returns=validation_returns.copy(deep=True),
            validation_context_returns=(validation_context_returns.copy(deep=True)),
        )

    @staticmethod
    def _copy_and_validate_dataset(
        dataset: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError(f"{dataset_name} must be a pandas DataFrame.")

        if dataset.empty:
            raise ValueError(f"{dataset_name} must not be empty.")

        required_columns = {
            "date",
            "ticker",
            "adjusted_close",
        }
        missing_columns = sorted(required_columns.difference(dataset.columns))

        if missing_columns:
            raise ValueError(
                f"{dataset_name} is missing required columns: {missing_columns}."
            )

        return dataset.copy(deep=True)

    @staticmethod
    def _extract_dates(
        dataset: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DatetimeIndex:
        try:
            dates = pd.DatetimeIndex(
                pd.to_datetime(
                    dataset["date"],
                    errors="raise",
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{dataset_name} must contain valid dates.") from error

        if dates.hasnans:
            raise ValueError(f"{dataset_name} must not contain missing dates.")

        return dates

    @staticmethod
    def _extract_tickers(
        dataset: pd.DataFrame,
        dataset_name: str,
    ) -> tuple[str, ...]:
        ticker_values = dataset["ticker"].tolist()

        if any(
            not isinstance(ticker, str) or not ticker.strip()
            for ticker in ticker_values
        ):
            raise ValueError(f"{dataset_name} must contain valid ticker names.")

        return tuple(sorted({ticker.strip() for ticker in ticker_values}))

    @staticmethod
    def _validate_split_boundaries(
        training_dates: pd.DatetimeIndex,
        validation_dates: pd.DatetimeIndex,
    ) -> None:
        training_max_date = pd.Timestamp(training_dates.to_numpy().max())
        validation_min_date = pd.Timestamp(validation_dates.to_numpy().min())

        if training_max_date >= validation_min_date:
            raise ValueError("Validation dates must begin after all training dates.")

    @staticmethod
    def _validate_ticker_sets(
        training_tickers: tuple[str, ...],
        validation_tickers: tuple[str, ...],
    ) -> None:
        if training_tickers != validation_tickers:
            missing_tickers = sorted(
                set(training_tickers).difference(validation_tickers)
            )
            unexpected_tickers = sorted(
                set(validation_tickers).difference(training_tickers)
            )
            raise ValueError(
                "Training and validation tickers must match "
                f"(missing: {missing_tickers}; "
                f"unexpected: {unexpected_tickers})."
            )

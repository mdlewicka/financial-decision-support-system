from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
)

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "adjusted_close",
    "volume",
    "daily_return",
)

RETURN_LAGS: tuple[int, ...] = (
    1,
    2,
    5,
    10,
    20,
)

TEMPORAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "return_lag_1",
    "return_lag_2",
    "return_lag_5",
    "return_lag_10",
    "return_lag_20",
    "return_mean_5",
    "return_mean_20",
    "return_volatility_5",
    "return_volatility_20",
    "price_momentum_5",
    "price_momentum_20",
    "price_to_previous_ma_5",
    "price_to_previous_ma_20",
    "rolling_drawdown_20",
    "rolling_drawdown_60",
    "volume_to_previous_mean_5",
    "volume_to_previous_mean_20",
    "market_return_lag_1",
    "market_return_mean_5",
    "market_return_volatility_20",
    "relative_strength_5",
    "relative_strength_20",
    "market_trend_regime",
    "market_volatility_regime",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    *IDENTIFIER_COLUMNS,
    *TEMPORAL_FEATURE_COLUMNS,
    "features_ready",
)


@dataclass(frozen=True, slots=True)
class TemporalFeatureContract:
    feature_columns: tuple[str, ...] = TEMPORAL_FEATURE_COLUMNS
    minimum_history_observations: int = 60
    uses_target_column: bool = False
    uses_future_observations: bool = False


class TemporalMarketFeatureTransformer:
    """Build backward-looking market features for Forecasting V2.

    The transformer stores only raw training history required to construct
    features at a later date. Target values are deliberately excluded from the
    contract. Every rolling statistic is trailing and every reference average
    used for a current observation is shifted by at least one observation.
    """

    contract = TemporalFeatureContract()

    def __init__(self) -> None:
        self._history: pd.DataFrame | None = None
        self._training_tickers: tuple[str, ...] = ()
        self._history_date_max: pd.Timestamp | None = None

    @property
    def is_fitted(self) -> bool:
        return self._history is not None

    @property
    def feature_names(self) -> tuple[str, ...]:
        return TEMPORAL_FEATURE_COLUMNS

    @property
    def training_tickers(self) -> tuple[str, ...]:
        return self._training_tickers

    @property
    def history_date_max(self) -> pd.Timestamp | None:
        return self._history_date_max

    def fit(
        self,
        training_dataset: pd.DataFrame,
    ) -> TemporalMarketFeatureTransformer:
        prepared = self._prepare_input(
            training_dataset,
            dataset_name="Training",
        )

        self._history = prepared
        self._training_tickers = tuple(
            sorted(prepared["ticker"].astype(str).unique().tolist())
        )
        self._history_date_max = pd.Timestamp(prepared["date"].max())
        return self

    def fit_transform(
        self,
        training_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        self.fit(training_dataset)

        if self._history is None:
            raise RuntimeError("Temporal feature history was not stored.")

        transformed = self._calculate_features(self._history)
        return transformed[list(OUTPUT_COLUMNS)].copy()

    def transform(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        if self._history is None or self._history_date_max is None:
            raise RuntimeError(
                "The temporal feature transformer must be fitted before transform()."
            )

        evaluation = self._prepare_input(
            dataset,
            dataset_name="Evaluation",
        )
        self._validate_evaluation(evaluation)

        history = self._history.assign(
            __segment="history",
            __original_index=-1,
        )
        current = evaluation.assign(
            __segment="evaluation",
            __original_index=np.arange(
                len(evaluation),
                dtype=int,
            ),
        )

        combined = pd.concat(
            [history, current],
            ignore_index=True,
        )

        transformed = self._calculate_features(combined)

        selected = transformed.loc[
            transformed["__segment"] == "evaluation"
        ].sort_values(
            "__original_index",
            kind="stable",
        )

        selected = selected.set_index("__original_index")
        selected.index = dataset.index

        return selected[list(OUTPUT_COLUMNS)].copy()

    @staticmethod
    def ready_rows(
        transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        if "features_ready" not in transformed.columns:
            raise ValueError("Transformed data must contain features_ready.")

        return transformed.loc[transformed["features_ready"]].copy()

    @staticmethod
    def _prepare_input(
        dataset: pd.DataFrame,
        dataset_name: str,
    ) -> pd.DataFrame:
        if dataset.empty:
            raise ValueError(f"{dataset_name} temporal feature data must not be empty.")

        missing_columns = sorted(
            set(REQUIRED_INPUT_COLUMNS).difference(dataset.columns)
        )

        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(
                f"{dataset_name} temporal feature data is missing columns: {joined}."
            )

        prepared = dataset[list(REQUIRED_INPUT_COLUMNS)].copy()

        prepared["date"] = pd.to_datetime(
            prepared["date"],
            errors="raise",
        )
        prepared["ticker"] = prepared["ticker"].astype("string")

        for column in (
            "adjusted_close",
            "volume",
            "daily_return",
        ):
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="raise",
            ).astype(float)

        if prepared[list(REQUIRED_INPUT_COLUMNS)].isna().any().any():
            raise ValueError(
                f"{dataset_name} temporal feature data contains missing values."
            )

        numeric_values = prepared[
            [
                "adjusted_close",
                "volume",
                "daily_return",
            ]
        ].to_numpy(dtype=float)

        if not np.isfinite(numeric_values).all():
            raise ValueError(
                f"{dataset_name} temporal feature data contains non-finite values."
            )

        if (prepared["adjusted_close"] <= 0.0).any():
            raise ValueError("Adjusted close values must be greater than zero.")

        if (prepared["volume"] < 0.0).any():
            raise ValueError("Volume values must be non-negative.")

        if prepared["ticker"].str.strip().eq("").any():
            raise ValueError("Ticker values must be non-empty.")

        if prepared.duplicated(["date", "ticker"]).any():
            raise ValueError(
                f"{dataset_name} temporal feature data contains "
                "duplicate date/ticker rows."
            )

        return prepared.sort_values(
            ["date", "ticker"],
            kind="stable",
            ignore_index=True,
        )

    def _validate_evaluation(
        self,
        evaluation: pd.DataFrame,
    ) -> None:
        if self._history_date_max is None:
            raise RuntimeError("Temporal feature transformer has no history boundary.")

        evaluation_date_min = pd.Timestamp(evaluation["date"].min())

        if evaluation_date_min <= self._history_date_max:
            raise ValueError(
                "Evaluation dates must begin after the fitted training history."
            )

        unknown_tickers = sorted(
            set(evaluation["ticker"].astype(str)).difference(self._training_tickers)
        )

        if unknown_tickers:
            raise ValueError(
                "Evaluation contains unknown tickers: "
                + ", ".join(unknown_tickers)
                + "."
            )

    @staticmethod
    def _calculate_features(
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        working = dataset.copy()

        working["__row_order"] = np.arange(
            len(working),
            dtype=int,
        )

        working = working.sort_values(
            ["ticker", "date"],
            kind="stable",
            ignore_index=True,
        )

        by_ticker = working.groupby(
            "ticker",
            sort=False,
            group_keys=False,
        )

        for lag in RETURN_LAGS:
            working[f"return_lag_{lag}"] = by_ticker["daily_return"].shift(lag)

        previous_returns = by_ticker["daily_return"].shift(1)

        previous_prices = by_ticker["adjusted_close"].shift(1)

        previous_volume = by_ticker["volume"].shift(1)

        working["return_mean_5"] = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_returns,
            window=5,
            operation="mean",
        )

        working["return_mean_20"] = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_returns,
            window=20,
            operation="mean",
        )

        working["return_volatility_5"] = (
            TemporalMarketFeatureTransformer._rolling_by_ticker(
                working,
                previous_returns,
                window=5,
                operation="std",
            )
        )

        working["return_volatility_20"] = (
            TemporalMarketFeatureTransformer._rolling_by_ticker(
                working,
                previous_returns,
                window=20,
                operation="std",
            )
        )

        working["price_momentum_5"] = (
            working["adjusted_close"] / by_ticker["adjusted_close"].shift(5) - 1.0
        )

        working["price_momentum_20"] = (
            working["adjusted_close"] / by_ticker["adjusted_close"].shift(20) - 1.0
        )

        previous_ma_5 = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_prices,
            window=5,
            operation="mean",
        )

        previous_ma_20 = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_prices,
            window=20,
            operation="mean",
        )

        working["price_to_previous_ma_5"] = (
            working["adjusted_close"] / previous_ma_5 - 1.0
        )

        working["price_to_previous_ma_20"] = (
            working["adjusted_close"] / previous_ma_20 - 1.0
        )

        rolling_peak_20 = by_ticker["adjusted_close"].transform(
            lambda values: values.rolling(
                20,
                min_periods=20,
            ).max()
        )

        rolling_peak_60 = by_ticker["adjusted_close"].transform(
            lambda values: values.rolling(
                60,
                min_periods=60,
            ).max()
        )

        working["rolling_drawdown_20"] = (
            working["adjusted_close"] / rolling_peak_20 - 1.0
        )

        working["rolling_drawdown_60"] = (
            working["adjusted_close"] / rolling_peak_60 - 1.0
        )

        previous_volume_mean_5 = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_volume,
            window=5,
            operation="mean",
        )

        previous_volume_mean_20 = TemporalMarketFeatureTransformer._rolling_by_ticker(
            working,
            previous_volume,
            window=20,
            operation="mean",
        )

        working["volume_to_previous_mean_5"] = (
            working["volume"] / previous_volume_mean_5 - 1.0
        )

        working["volume_to_previous_mean_20"] = (
            working["volume"] / previous_volume_mean_20 - 1.0
        )

        market_features = TemporalMarketFeatureTransformer._build_market_features(
            working
        )

        working = working.merge(
            market_features,
            on="date",
            how="left",
            validate="many_to_one",
        )

        working["relative_strength_5"] = (
            working["price_momentum_5"] - working["market_compound_return_5"]
        )

        working["relative_strength_20"] = (
            working["price_momentum_20"] - working["market_compound_return_20"]
        )

        working = working.drop(
            columns=[
                "market_compound_return_5",
                "market_compound_return_20",
            ]
        )

        feature_values = working[list(TEMPORAL_FEATURE_COLUMNS)].to_numpy(
            dtype=float,
        )

        features_ready = pd.Series(
            np.asarray(
                np.isfinite(feature_values).all(axis=1),
                dtype=bool,
            ),
            index=working.index,
            dtype="bool",
        )

        working["features_ready"] = features_ready

        ready_values = working.loc[
            features_ready,
            list(TEMPORAL_FEATURE_COLUMNS),
        ].to_numpy(dtype=float)

        if ready_values.size and not np.isfinite(ready_values).all():
            raise RuntimeError("Temporal feature calculation produced invalid values.")

        return working.sort_values(
            "__row_order",
            kind="stable",
            ignore_index=True,
        )

    @staticmethod
    def _rolling_by_ticker(
        working: pd.DataFrame,
        values: pd.Series,
        window: int,
        operation: str,
    ) -> pd.Series:
        temporary = pd.DataFrame(
            {
                "ticker": working["ticker"],
                "value": values,
            },
            index=working.index,
        )

        rolling = temporary.groupby(
            "ticker",
            sort=False,
        )["value"].rolling(
            window,
            min_periods=window,
        )

        if operation == "mean":
            result = rolling.mean()
        elif operation == "std":
            result = rolling.std(ddof=1)
        else:
            raise ValueError(f"Unsupported rolling operation: {operation}.")

        return result.reset_index(
            level=0,
            drop=True,
        ).sort_index()

    @staticmethod
    def _build_market_features(
        working: pd.DataFrame,
    ) -> pd.DataFrame:
        market = (
            working.groupby(
                "date",
                sort=True,
            )["daily_return"]
            .mean()
            .rename("market_return")
            .reset_index()
        )

        previous_market_return = market["market_return"].shift(1)

        market["market_return_lag_1"] = previous_market_return

        market["market_return_mean_5"] = previous_market_return.rolling(
            5,
            min_periods=5,
        ).mean()

        market["market_return_volatility_20"] = previous_market_return.rolling(
            20,
            min_periods=20,
        ).std(ddof=1)

        market["market_compound_return_5"] = (1.0 + market["market_return"]).rolling(
            5,
            min_periods=5,
        ).apply(
            np.prod,
            raw=True,
        ) - 1.0

        market["market_compound_return_20"] = (1.0 + market["market_return"]).rolling(
            20,
            min_periods=20,
        ).apply(
            np.prod,
            raw=True,
        ) - 1.0

        market["market_trend_regime"] = np.sign(
            market["market_return_mean_5"].to_numpy(dtype=float)
        ).astype(float)

        historical_volatility_threshold = (
            market["market_return_volatility_20"]
            .shift(1)
            .expanding(min_periods=20)
            .median()
        )

        volatility_available = historical_volatility_threshold.notna().to_numpy(
            dtype=bool,
        )

        volatility_values = market["market_return_volatility_20"].to_numpy(
            dtype=float,
        )
        threshold_values = historical_volatility_threshold.to_numpy(
            dtype=float,
        )

        high_volatility = np.greater(
            volatility_values,
            threshold_values,
        )

        volatility_regime = np.where(
            volatility_available,
            high_volatility.astype(float),
            np.nan,
        )

        market["market_volatility_regime"] = pd.Series(
            np.asarray(volatility_regime, dtype=float),
            index=market.index,
            dtype="float64",
        )

        return market.drop(columns=["market_return"])

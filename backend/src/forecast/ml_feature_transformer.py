import numpy as np
import pandas as pd

CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = ("ticker",)

DIRECT_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "daily_return",
    "volume_change",
    "volatility_5",
    "volatility_20",
    "momentum_5",
    "momentum_20",
)

DERIVED_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "adjusted_close_to_ma_5",
    "adjusted_close_to_ma_20",
    "intraday_range",
    "open_to_close_return",
)

NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    *DIRECT_NUMERIC_FEATURE_COLUMNS,
    *DERIVED_NUMERIC_FEATURE_COLUMNS,
)

MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    *CATEGORICAL_FEATURE_COLUMNS,
    *NUMERIC_FEATURE_COLUMNS,
)

REQUIRED_NUMERIC_INPUT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "daily_return",
    "volume_change",
    "moving_average_5",
    "moving_average_20",
    "volatility_5",
    "volatility_20",
    "momentum_5",
    "momentum_20",
)

REQUIRED_INPUT_COLUMNS = frozenset(
    (
        "ticker",
        *REQUIRED_NUMERIC_INPUT_COLUMNS,
    )
)

POSITIVE_PRICE_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "moving_average_5",
    "moving_average_20",
)


class MarketMLFeatureTransformer:
    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        self._validate_input_data(data)

        numeric_input = pd.DataFrame(
            {
                column: pd.to_numeric(
                    data[column],
                    errors="raise",
                )
                for column in REQUIRED_NUMERIC_INPUT_COLUMNS
            },
            index=data.index,
        )

        self._validate_numeric_input(numeric_input)

        transformed = pd.DataFrame(
            index=data.index,
        )
        transformed["ticker"] = data["ticker"].astype(str)

        for column in DIRECT_NUMERIC_FEATURE_COLUMNS:
            transformed[column] = numeric_input[column].astype(float)

        transformed["adjusted_close_to_ma_5"] = (
            numeric_input["adjusted_close"] / numeric_input["moving_average_5"] - 1.0
        )
        transformed["adjusted_close_to_ma_20"] = (
            numeric_input["adjusted_close"] / numeric_input["moving_average_20"] - 1.0
        )
        transformed["intraday_range"] = (
            numeric_input["high"] - numeric_input["low"]
        ) / numeric_input["close"]
        transformed["open_to_close_return"] = (
            numeric_input["close"] / numeric_input["open"] - 1.0
        )

        self._validate_transformed_features(transformed)

        return transformed[list(MODEL_FEATURE_COLUMNS)].copy()

    @staticmethod
    def _validate_input_data(
        data: pd.DataFrame,
    ) -> None:
        if data.empty:
            raise ValueError("ML feature input data must not be empty.")

        missing_columns = sorted(REQUIRED_INPUT_COLUMNS.difference(data.columns))

        if missing_columns:
            joined_columns = ", ".join(missing_columns)
            raise ValueError(f"Missing required ML feature columns: {joined_columns}.")

        tickers = data["ticker"].astype("string")

        if tickers.isna().any() or tickers.str.strip().eq("").any():
            raise ValueError("Ticker values must be non-empty.")

    @staticmethod
    def _validate_numeric_input(
        numeric_input: pd.DataFrame,
    ) -> None:
        numeric_values = numeric_input.to_numpy(
            dtype=float,
        )

        if not np.isfinite(numeric_values).all():
            raise ValueError("ML feature input contains non-finite values.")

        non_positive_columns = [
            column
            for column in POSITIVE_PRICE_COLUMNS
            if (numeric_input[column] <= 0.0).any()
        ]

        if non_positive_columns:
            joined_columns = ", ".join(non_positive_columns)
            raise ValueError(
                "ML feature input contains non-positive "
                f"price values in: {joined_columns}."
            )

        if (numeric_input["high"] < numeric_input["low"]).any():
            raise ValueError("High price must not be lower than low price.")

    @staticmethod
    def _validate_transformed_features(
        transformed: pd.DataFrame,
    ) -> None:
        numeric_features = transformed[list(NUMERIC_FEATURE_COLUMNS)].to_numpy(
            dtype=float,
        )

        if not np.isfinite(numeric_features).all():
            raise RuntimeError("ML feature transformation produced non-finite values.")

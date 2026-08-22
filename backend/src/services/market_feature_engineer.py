import numpy as np
import pandas as pd


class MarketFeatureEngineer:
    """Tworzy cechy finansowe na podstawie danych rynkowych."""

    def engineer(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        featured_data = data.copy()

        featured_data = featured_data.sort_values(
            by=["ticker", "date"],
        ).reset_index(drop=True)

        grouped_data = featured_data.groupby(
            "ticker",
            sort=False,
        )

        featured_data["daily_return"] = grouped_data["adjusted_close"].pct_change()

        featured_data["log_return"] = grouped_data["adjusted_close"].transform(
            lambda prices: np.log(prices / prices.shift(1))
        )

        featured_data["price_change"] = grouped_data["adjusted_close"].diff()

        featured_data["volume_change"] = grouped_data["volume"].pct_change()

        featured_data["moving_average_5"] = grouped_data["adjusted_close"].transform(
            lambda prices: prices.rolling(
                window=5,
            ).mean()
        )

        featured_data["moving_average_20"] = grouped_data["adjusted_close"].transform(
            lambda prices: prices.rolling(
                window=20,
            ).mean()
        )

        featured_data["volatility_5"] = grouped_data["adjusted_close"].transform(
            lambda prices: prices.pct_change().rolling(window=5).std()
        )

        featured_data["volatility_20"] = grouped_data["adjusted_close"].transform(
            lambda prices: prices.pct_change().rolling(window=20).std()
        )

        featured_data["momentum_5"] = grouped_data["adjusted_close"].transform(
            lambda prices: (prices / prices.shift(5)) - 1
        )

        featured_data["momentum_20"] = grouped_data["adjusted_close"].transform(
            lambda prices: (prices / prices.shift(20)) - 1
        )

        return featured_data

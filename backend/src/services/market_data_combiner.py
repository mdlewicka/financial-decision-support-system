import pandas as pd


class MarketDataCombiner:
    def combine(
        self,
        data_frames: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if not data_frames:
            raise ValueError("At least one DataFrame must be provided.")

        combined_data = pd.concat(
            data_frames.values(),
            ignore_index=True,
        )

        combined_data = combined_data.sort_values(
            by=["ticker", "date"],
        )

        combined_data = combined_data.reset_index(
            drop=True,
        )

        return combined_data

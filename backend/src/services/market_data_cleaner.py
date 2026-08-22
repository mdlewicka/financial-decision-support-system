import pandas as pd


class MarketDataCleaner:
    def clean(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        cleaned_data = data.copy()

        cleaned_data = cleaned_data.dropna(how="all")

        cleaned_data = cleaned_data.drop_duplicates()

        cleaned_data = cleaned_data.sort_values(
            by=["ticker", "date"],
        )

        cleaned_data = cleaned_data.reset_index(drop=True)

        return cleaned_data

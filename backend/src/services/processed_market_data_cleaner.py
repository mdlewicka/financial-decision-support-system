import numpy as np
import pandas as pd


class ProcessedMarketDataCleaner:
    """Czyści dane po wygenerowaniu cech."""

    def clean(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        cleaned_data = data.copy()

        cleaned_data = cleaned_data.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        cleaned_data = cleaned_data.dropna()

        cleaned_data = cleaned_data.drop_duplicates()

        cleaned_data = cleaned_data.sort_values(
            by=[
                "ticker",
                "date",
            ],
        )

        cleaned_data = cleaned_data.reset_index(
            drop=True,
        )

        return cleaned_data

import pandas as pd


class MarketDatasetBuilder:
    def build(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        dataset = data.copy()

        dataset = dataset.sort_values(
            by=[
                "ticker",
                "date",
            ],
        )

        dataset = dataset.reset_index(
            drop=True,
        )

        return dataset

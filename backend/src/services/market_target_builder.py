import pandas as pd


class MarketTargetBuilder:
    """Buduje target następnej stopy zwrotu osobno dla każdego instrumentu."""

    def build(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        dataset = data.copy()

        grouped_dataset = dataset.groupby(
            "ticker",
            sort=False,
        )

        next_adjusted_close = grouped_dataset["adjusted_close"].shift(-1)

        dataset["target_date"] = grouped_dataset["date"].shift(-1)

        dataset["target"] = (next_adjusted_close / dataset["adjusted_close"]) - 1.0

        return dataset

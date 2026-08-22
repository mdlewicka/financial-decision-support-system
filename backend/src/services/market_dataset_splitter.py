import pandas as pd


class MarketDatasetSplitter:
    TRAIN_RATIO = 0.70
    VALIDATION_RATIO = 0.15

    REQUIRED_COLUMNS = frozenset(
        {
            "ticker",
            "date",
            "target_date",
        }
    )

    def split(
        self,
        data: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        dataset = data.copy()

        self._validate_required_columns(dataset)
        self._validate_target_dates(dataset)

        ordered_target_dates = (
            dataset["target_date"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        number_of_target_dates = len(ordered_target_dates)

        train_end = int(number_of_target_dates * self.TRAIN_RATIO)
        validation_end = int(
            number_of_target_dates * (self.TRAIN_RATIO + self.VALIDATION_RATIO)
        )

        self._validate_split_sizes(
            number_of_target_dates=number_of_target_dates,
            train_end=train_end,
            validation_end=validation_end,
        )

        train_cutoff = ordered_target_dates.iloc[train_end - 1]
        validation_cutoff = ordered_target_dates.iloc[validation_end - 1]

        train_dataset = dataset.loc[dataset["target_date"] <= train_cutoff]
        validation_dataset = dataset.loc[
            (dataset["target_date"] > train_cutoff)
            & (dataset["target_date"] <= validation_cutoff)
        ]
        test_dataset = dataset.loc[dataset["target_date"] > validation_cutoff]

        train_dataset = self._sort_dataset(train_dataset)
        validation_dataset = self._sort_dataset(validation_dataset)
        test_dataset = self._sort_dataset(test_dataset)

        self._validate_shared_boundaries(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

        return (
            train_dataset,
            validation_dataset,
            test_dataset,
        )

    def _validate_required_columns(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        missing_columns = sorted(self.REQUIRED_COLUMNS.difference(dataset.columns))

        if missing_columns:
            joined_columns = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {joined_columns}")

    @staticmethod
    def _validate_target_dates(
        dataset: pd.DataFrame,
    ) -> None:
        if dataset[["date", "target_date"]].isna().any().any():
            raise ValueError("date and target_date must not contain missing values.")

        if (dataset["target_date"] <= dataset["date"]).any():
            raise ValueError("target_date must be later than date.")

    @staticmethod
    def _validate_split_sizes(
        number_of_target_dates: int,
        train_end: int,
        validation_end: int,
    ) -> None:
        invalid_split = (
            train_end <= 0
            or validation_end <= train_end
            or validation_end >= number_of_target_dates
        )

        if invalid_split:
            raise ValueError(
                "Not enough distinct target dates to create "
                "train, validation and test datasets."
            )

    @staticmethod
    def _sort_dataset(
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        return dataset.sort_values(
            by=[
                "ticker",
                "date",
            ],
        ).reset_index(
            drop=True,
        )

    @staticmethod
    def _validate_shared_boundaries(
        train_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
    ) -> None:
        if train_dataset.empty or validation_dataset.empty or test_dataset.empty:
            raise ValueError("Train, validation and test datasets must not be empty.")

        train_target_max = train_dataset["target_date"].max()
        validation_target_min = validation_dataset["target_date"].min()
        validation_target_max = validation_dataset["target_date"].max()
        test_target_min = test_dataset["target_date"].min()

        boundaries_overlap = (
            train_target_max >= validation_target_min
            or validation_target_max >= test_target_min
        )

        if boundaries_overlap:
            raise ValueError("Target-date ranges of dataset splits must not overlap.")

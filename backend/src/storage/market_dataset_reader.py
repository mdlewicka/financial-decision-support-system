from pathlib import Path

import pandas as pd

from backend.src.config import (
    MARKET_DATASETS_DIR,
    TEST_DATASET_FILENAME,
    TRAIN_DATASET_FILENAME,
    VALIDATION_DATASET_FILENAME,
)


class MarketDatasetReader:
    DATE_COLUMNS = [
        "date",
        "target_date",
    ]

    def __init__(
        self,
        input_directory: Path = MARKET_DATASETS_DIR,
    ) -> None:
        self.input_directory = input_directory

    def read(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_dataset, validation_dataset = self.read_training_and_validation()
        test_dataset = self._read_dataset(TEST_DATASET_FILENAME)

        return (
            train_dataset,
            validation_dataset,
            test_dataset,
        )

    def read_training_and_validation(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_dataset = self._read_dataset(TRAIN_DATASET_FILENAME)
        validation_dataset = self._read_dataset(VALIDATION_DATASET_FILENAME)

        return (
            train_dataset,
            validation_dataset,
        )

    def _read_dataset(
        self,
        file_name: str,
    ) -> pd.DataFrame:
        return pd.read_csv(
            self.input_directory / file_name,
            parse_dates=self.DATE_COLUMNS,
        )

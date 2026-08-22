from pathlib import Path

import pandas as pd

from backend.src.config import (
    MARKET_DATASETS_DIR,
    TEST_DATASET_FILENAME,
    TRAIN_DATASET_FILENAME,
    VALIDATION_DATASET_FILENAME,
)


class MarketDatasetWriter:
    def __init__(
        self,
        output_directory: Path = MARKET_DATASETS_DIR,
    ) -> None:
        self.output_directory = output_directory

    def write(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> tuple[Path, Path, Path]:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = self.output_directory / TRAIN_DATASET_FILENAME
        validation_path = self.output_directory / VALIDATION_DATASET_FILENAME
        test_path = self.output_directory / TEST_DATASET_FILENAME

        train_data.to_csv(
            train_path,
            index=False,
        )
        validation_data.to_csv(
            validation_path,
            index=False,
        )
        test_data.to_csv(
            test_path,
            index=False,
        )

        return (
            train_path,
            validation_path,
            test_path,
        )

from pathlib import Path

import pandas as pd

from backend.src.services.market_dataset_builder import MarketDatasetBuilder
from backend.src.services.market_dataset_splitter import MarketDatasetSplitter
from backend.src.services.market_target_builder import MarketTargetBuilder
from backend.src.storage.market_dataset_writer import MarketDatasetWriter


class MarketDatasetPreparationService:
    def __init__(
        self,
        dataset_builder: MarketDatasetBuilder,
        target_builder: MarketTargetBuilder,
        dataset_splitter: MarketDatasetSplitter,
        dataset_writer: MarketDatasetWriter,
    ) -> None:
        self.dataset_builder = dataset_builder
        self.target_builder = target_builder
        self.dataset_splitter = dataset_splitter
        self.dataset_writer = dataset_writer

    def prepare(
        self,
        data: pd.DataFrame,
    ) -> tuple[Path, Path, Path]:
        dataset = self.dataset_builder.build(data)

        dataset_with_target = self.target_builder.build(dataset)

        complete_dataset = dataset_with_target.dropna(
            subset=[
                "target",
                "target_date",
            ],
        ).reset_index(
            drop=True,
        )

        (
            train_data,
            validation_data,
            test_data,
        ) = self.dataset_splitter.split(complete_dataset)

        return self.dataset_writer.write(
            train_data=train_data,
            validation_data=validation_data,
            test_data=test_data,
        )

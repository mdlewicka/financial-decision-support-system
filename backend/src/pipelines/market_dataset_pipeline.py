from pathlib import Path

import pandas as pd

from backend.src.services.market_dataset_preparation_service import (
    MarketDatasetPreparationService,
)


class MarketDatasetPipeline:
    def __init__(
        self,
        preparation_service: MarketDatasetPreparationService,
    ) -> None:
        self.preparation_service = preparation_service

    def run(
        self,
        processed_data_path: Path,
    ) -> tuple[Path, Path, Path]:
        processed_data = pd.read_csv(
            processed_data_path,
            parse_dates=["date"],
        )

        return self.preparation_service.prepare(processed_data)

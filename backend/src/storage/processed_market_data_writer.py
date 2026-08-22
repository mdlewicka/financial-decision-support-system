from pathlib import Path

import pandas as pd

from backend.src.config import (
    MARKET_FEATURES_FILENAME,
    PROCESSED_MARKET_DATA_DIR,
)


class ProcessedMarketDataWriter:
    """Zapisuje dane z cechami do warstwy processed."""

    def __init__(
        self,
        output_directory: Path = PROCESSED_MARKET_DATA_DIR,
    ) -> None:
        self._output_directory = output_directory

    def write(
        self,
        data: pd.DataFrame,
        file_name: str = MARKET_FEATURES_FILENAME,
    ) -> Path:
        if data.empty:
            raise ValueError("Cannot write an empty DataFrame.")

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = self._output_directory / file_name

        data.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d",
        )

        return output_path

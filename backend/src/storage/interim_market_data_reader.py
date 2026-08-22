from pathlib import Path

import pandas as pd

from backend.src.config import (
    COMBINED_MARKET_DATA_FILENAME,
    DATE_COLUMN_NAME,
    INTERIM_MARKET_DATA_DIR,
)


class InterimMarketDataReader:
    """Odczytuje połączone dane rynkowe z warstwy interim."""

    def __init__(
        self,
        data_directory: Path = INTERIM_MARKET_DATA_DIR,
    ) -> None:
        self._data_directory = data_directory

    def read(
        self,
        file_name: str = COMBINED_MARKET_DATA_FILENAME,
    ) -> pd.DataFrame:
        file_path = self._data_directory / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Market data file not found: {file_path}")

        data = pd.read_csv(
            file_path,
            parse_dates=[DATE_COLUMN_NAME],
        )

        return data

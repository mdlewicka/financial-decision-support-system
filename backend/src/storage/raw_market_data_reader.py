from pathlib import Path

import pandas as pd

from backend.src.config import (
    DATE_COLUMN_NAME,
    RAW_DATA_FILE_FORMAT,
    RAW_MARKET_DATA_DIR,
)


class RawMarketDataReader:
    def __init__(
        self,
        data_directory: Path = RAW_MARKET_DATA_DIR,
    ) -> None:
        self._data_directory = data_directory

    def read(self, ticker: str) -> pd.DataFrame:
        file_path = self._build_file_path(ticker)

        if not file_path.exists():
            raise FileNotFoundError(f"Market data file not found: {file_path}")

        data = pd.read_csv(
            file_path,
            parse_dates=[DATE_COLUMN_NAME],
        )

        return data

    def _build_file_path(self, ticker: str) -> Path:
        return self._data_directory / f"{ticker}.{RAW_DATA_FILE_FORMAT}"

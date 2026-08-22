from pathlib import Path

import pandas as pd

from backend.src.config import RAW_MARKET_DATA_DIR
from backend.src.storage.exceptions import MarketDataWriteError


class RawMarketDataWriter:
    """Zapisuje zweryfikowane dane rynkowe w warstwie raw."""

    def __init__(
        self,
        output_directory: Path = RAW_MARKET_DATA_DIR,
    ) -> None:
        self._output_directory = output_directory

    def save(
        self,
        data: pd.DataFrame,
        ticker: str,
    ) -> Path:
        """Zapisuje dane instrumentu do pliku CSV i zwraca jego ścieżkę."""

        normalized_ticker = self._normalize_ticker(ticker)
        self._validate_data(data)

        output_path = self._output_directory / f"{normalized_ticker}.csv"
        temporary_path = output_path.with_suffix(".tmp")

        try:
            self._output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            data.to_csv(
                temporary_path,
                index=False,
                date_format="%Y-%m-%d",
            )

            temporary_path.replace(output_path)

        except (OSError, ValueError) as error:
            self._remove_temporary_file(temporary_path)

            raise MarketDataWriteError(
                "Nie udało się zapisać danych rynkowych "
                f"dla tickera {normalized_ticker!r} "
                f"do pliku {output_path}."
            ) from error

        return output_path

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        normalized_ticker = ticker.strip().upper()

        if not normalized_ticker:
            raise MarketDataWriteError("Ticker używany do zapisu nie może być pusty.")

        return normalized_ticker

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        if data is None or data.empty:
            raise MarketDataWriteError("Nie można zapisać pustych danych rynkowych.")

    @staticmethod
    def _remove_temporary_file(temporary_path: Path) -> None:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

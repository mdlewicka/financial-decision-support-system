from datetime import date

import pandas as pd
import yfinance as yf

from backend.src.providers.exceptions import (
    EmptyMarketDataError,
    MarketDataDownloadError,
)


class YahooFinanceProvider:
    """Pobiera historyczne dane rynkowe z Yahoo Finance."""

    def __init__(self, timeout_seconds: int = 30) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Timeout musi być większy od zera.")

        self._timeout_seconds = timeout_seconds

    def fetch_history(
        self,
        ticker: str,
        start_date: date,
        end_date: date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Pobiera surowe dane historyczne dla jednego instrumentu."""

        normalized_ticker = ticker.strip().upper()

        self._validate_parameters(
            ticker=normalized_ticker,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )

        try:
            data = yf.download(
                tickers=normalized_ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
                timeout=self._timeout_seconds,
                multi_level_index=False,
            )
        except Exception as error:
            raise MarketDataDownloadError(
                f"Nie udało się pobrać danych dla tickera {normalized_ticker!r}."
            ) from error

        if data is None or data.empty:
            raise EmptyMarketDataError(
                f"Yahoo Finance nie zwróciło danych dla tickera "
                f"{normalized_ticker!r} w zakresie od {start_date}"
                f"{f' do {end_date}' if end_date else ''}."
            )

        return data.copy()

    @staticmethod
    def _validate_parameters(
        ticker: str,
        start_date: date,
        end_date: date | None,
        interval: str,
    ) -> None:
        if not ticker:
            raise ValueError("Ticker nie może być pusty.")

        if not interval.strip():
            raise ValueError("Interwał nie może być pusty.")

        if end_date is not None and end_date <= start_date:
            raise ValueError("Data końcowa musi być późniejsza od daty początkowej.")

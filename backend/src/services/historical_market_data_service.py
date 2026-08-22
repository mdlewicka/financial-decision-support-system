from collections.abc import Iterable
from datetime import date
from pathlib import Path

from backend.src.config import SUPPORTED_TICKERS
from backend.src.providers.yahoo_finance import YahooFinanceProvider
from backend.src.services.market_data_batch_result import (
    MarketDataBatchResult,
    MarketDataDownloadFailure,
)
from backend.src.services.market_data_standardizer import (
    MarketDataStandardizer,
)
from backend.src.services.market_data_validator import (
    MarketDataValidator,
)
from backend.src.storage.raw_market_data_writer import (
    RawMarketDataWriter,
)
from backend.src.utils.ticker_utils import (
    normalize_tickers,
)


class HistoricalMarketDataService:
    """Pobiera, standaryzuje, waliduje i zapisuje dane historyczne."""

    def __init__(
        self,
        provider: YahooFinanceProvider | None = None,
        standardizer: MarketDataStandardizer | None = None,
        validator: MarketDataValidator | None = None,
        writer: RawMarketDataWriter | None = None,
    ) -> None:
        self._provider = provider or YahooFinanceProvider()
        self._standardizer = standardizer or MarketDataStandardizer()
        self._validator = validator or MarketDataValidator()
        self._writer = writer or RawMarketDataWriter()

    def download_history(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> Path:
        """Pobiera dane historyczne i zapisuje je na dysku."""

        raw_data = self._provider.fetch_history(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        standardized_data = self._standardizer.standardize(
            data=raw_data,
            ticker=ticker,
        )

        validated_data = self._validator.validate(
            data=standardized_data,
            expected_ticker=ticker,
        )

        return self._writer.save(
            data=validated_data,
            ticker=ticker,
        )

    def download_many(
        self,
        tickers: Iterable[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, Path]:
        """Pobiera i zapisuje dane historyczne dla wielu tickerów."""

        normalized_tickers = normalize_tickers(tickers)

        return {
            ticker: self.download_history(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
            )
            for ticker in normalized_tickers
        }

    def download_many_safe(
        self,
        tickers: Iterable[str],
        start_date: date,
        end_date: date,
    ) -> MarketDataBatchResult:
        """
        Pobiera dane dla wielu tickerów bez zatrzymywania procesu
        po błędzie pojedynczego instrumentu.
        """

        normalized_tickers = normalize_tickers(tickers)

        successful: dict[str, Path] = {}
        failed: dict[str, MarketDataDownloadFailure] = {}

        for ticker in normalized_tickers:
            try:
                saved_path = self.download_history(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                )
            except Exception as error:
                failed[ticker] = MarketDataDownloadFailure(
                    ticker=ticker,
                    error_type=type(error).__name__,
                    message=str(error),
                )
            else:
                successful[ticker] = saved_path

        return MarketDataBatchResult(
            successful=successful,
            failed=failed,
        )

    def download_supported_history(
        self,
        start_date: date,
        end_date: date,
    ) -> dict[str, Path]:
        """Pobiera dane dla wszystkich tickerów z konfiguracji."""

        return self.download_many(
            tickers=SUPPORTED_TICKERS,
            start_date=start_date,
            end_date=end_date,
        )

    def download_supported_history_safe(
        self,
        start_date: date,
        end_date: date,
    ) -> MarketDataBatchResult:
        """Pobiera skonfigurowane tickery i zwraca zbiorczy raport."""

        return self.download_many_safe(
            tickers=SUPPORTED_TICKERS,
            start_date=start_date,
            end_date=end_date,
        )

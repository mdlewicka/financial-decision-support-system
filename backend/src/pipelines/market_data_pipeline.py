from collections.abc import Iterable
from datetime import date
from pathlib import Path

from backend.src.services.historical_market_data_service import (
    HistoricalMarketDataService,
)
from backend.src.services.market_data_processing_service import (
    MarketDataProcessingService,
)
from backend.src.services.market_feature_processing_service import (
    MarketFeatureProcessingService,
)
from backend.src.utils.ticker_utils import (
    normalize_tickers,
)


class MarketDataPipeline:
    """Pobiera i przetwarza historyczne dane rynkowe."""

    def __init__(
        self,
        historical_data_service: HistoricalMarketDataService | None = None,
        processing_service: MarketDataProcessingService | None = None,
        feature_processing_service: (MarketFeatureProcessingService | None) = None,
    ) -> None:
        self._historical_data_service = (
            historical_data_service or HistoricalMarketDataService()
        )

        self._processing_service = processing_service or MarketDataProcessingService()

        self._feature_processing_service = (
            feature_processing_service or MarketFeatureProcessingService()
        )

    def run(
        self,
        tickers: Iterable[str],
        start_date: date,
        end_date: date,
    ) -> Path:
        """Pobiera dane do warstwy raw, a następnie zapisuje warstwę interim."""

        normalized_tickers = normalize_tickers(tickers)

        self._historical_data_service.download_many(
            tickers=normalized_tickers,
            start_date=start_date,
            end_date=end_date,
        )

        self._processing_service.process(list(normalized_tickers))

        return self._feature_processing_service.process()

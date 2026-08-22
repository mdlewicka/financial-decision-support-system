from pathlib import Path

from backend.src.services.market_feature_engineer import (
    MarketFeatureEngineer,
)
from backend.src.services.processed_market_data_cleaner import (
    ProcessedMarketDataCleaner,
)
from backend.src.storage.interim_market_data_reader import (
    InterimMarketDataReader,
)
from backend.src.storage.processed_market_data_writer import (
    ProcessedMarketDataWriter,
)


class MarketFeatureProcessingService:
    """Przetwarza dane interim do warstwy processed."""

    def __init__(
        self,
        reader: InterimMarketDataReader | None = None,
        feature_engineer: MarketFeatureEngineer | None = None,
        cleaner: ProcessedMarketDataCleaner | None = None,
        writer: ProcessedMarketDataWriter | None = None,
    ) -> None:
        self._reader = reader or InterimMarketDataReader()
        self._feature_engineer = feature_engineer or MarketFeatureEngineer()
        self._cleaner = cleaner or ProcessedMarketDataCleaner()
        self._writer = writer or ProcessedMarketDataWriter()

    def process(self) -> Path:
        market_data = self._reader.read()

        featured_data = self._feature_engineer.engineer(market_data)

        cleaned_data = self._cleaner.clean(featured_data)

        return self._writer.write(cleaned_data)

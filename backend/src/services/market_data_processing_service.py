from pathlib import Path

import pandas as pd

from backend.src.services.market_data_cleaner import (
    MarketDataCleaner,
)
from backend.src.services.market_data_combiner import (
    MarketDataCombiner,
)
from backend.src.storage.interim_market_data_writer import (
    InterimMarketDataWriter,
)
from backend.src.storage.raw_market_data_reader import (
    RawMarketDataReader,
)


class MarketDataProcessingService:
    def __init__(
        self,
        reader: RawMarketDataReader | None = None,
        combiner: MarketDataCombiner | None = None,
        cleaner: MarketDataCleaner | None = None,
        writer: InterimMarketDataWriter | None = None,
    ) -> None:
        self._reader = reader or RawMarketDataReader()
        self._combiner = combiner or MarketDataCombiner()
        self._cleaner = cleaner or MarketDataCleaner()
        self._writer = writer or InterimMarketDataWriter()

    def process(
        self,
        tickers: list[str],
    ) -> Path:
        market_data = self._read_market_data(tickers)

        combined_data = self._combiner.combine(market_data)

        cleaned_data = self._cleaner.clean(combined_data)

        return self._writer.write(cleaned_data)

    def _read_market_data(
        self,
        tickers: list[str],
    ) -> dict[str, pd.DataFrame]:
        return {ticker: self._reader.read(ticker) for ticker in tickers}

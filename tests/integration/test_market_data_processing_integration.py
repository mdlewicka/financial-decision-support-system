from pathlib import Path

import pandas as pd

from backend.src.services.market_data_cleaner import (
    MarketDataCleaner,
)
from backend.src.services.market_data_combiner import (
    MarketDataCombiner,
)
from backend.src.services.market_data_processing_service import (
    MarketDataProcessingService,
)
from backend.src.storage.interim_market_data_writer import (
    InterimMarketDataWriter,
)
from backend.src.storage.raw_market_data_reader import (
    RawMarketDataReader,
)


def test_market_data_processing_pipeline(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    interim_directory = tmp_path / "interim"

    raw_directory.mkdir()

    spy_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "SPY",
                "SPY",
            ],
            "open": [590.0, 591.0],
            "high": [595.0, 596.0],
            "low": [589.0, 590.0],
            "close": [593.0, 594.0],
            "adjusted_close": [593.0, 594.0],
            "volume": [1000, 1100],
        }
    )

    qqq_data = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "QQQ",
                "QQQ",
            ],
            "open": [500.0, 501.0],
            "high": [505.0, 506.0],
            "low": [499.0, 500.0],
            "close": [503.0, 504.0],
            "adjusted_close": [503.0, 504.0],
            "volume": [2000, 2100],
        }
    )

    spy_data.to_csv(
        raw_directory / "SPY.csv",
        index=False,
    )

    qqq_data.to_csv(
        raw_directory / "QQQ.csv",
        index=False,
    )

    service = MarketDataProcessingService(
        reader=RawMarketDataReader(raw_directory),
        combiner=MarketDataCombiner(),
        cleaner=MarketDataCleaner(),
        writer=InterimMarketDataWriter(interim_directory),
    )

    output_path = service.process(
        [
            "SPY",
            "QQQ",
        ]
    )

    assert output_path.exists()

    result = pd.read_csv(
        output_path,
        parse_dates=["date"],
    )

    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "ticker": [
                "QQQ",
                "QQQ",
                "SPY",
                "SPY",
            ],
            "open": [
                500.0,
                501.0,
                590.0,
                591.0,
            ],
            "high": [
                505.0,
                506.0,
                595.0,
                596.0,
            ],
            "low": [
                499.0,
                500.0,
                589.0,
                590.0,
            ],
            "close": [
                503.0,
                504.0,
                593.0,
                594.0,
            ],
            "adjusted_close": [
                503.0,
                504.0,
                593.0,
                594.0,
            ],
            "volume": [
                2000,
                2100,
                1000,
                1100,
            ],
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )

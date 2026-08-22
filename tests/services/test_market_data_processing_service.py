from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from backend.src.services.market_data_processing_service import (
    MarketDataProcessingService,
)


def test_process_runs_complete_pipeline() -> None:
    reader = Mock()
    combiner = Mock()
    cleaner = Mock()
    writer = Mock()

    raw_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    combined_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    cleaned_data = pd.DataFrame(
        {
            "ticker": ["SPY"],
            "close": [593.0],
        }
    )

    output_path = Path("market_data.csv")

    reader.read.return_value = raw_data
    combiner.combine.return_value = combined_data
    cleaner.clean.return_value = cleaned_data
    writer.write.return_value = output_path

    service = MarketDataProcessingService(
        reader=reader,
        combiner=combiner,
        cleaner=cleaner,
        writer=writer,
    )

    result = service.process(["SPY"])

    reader.read.assert_called_once_with("SPY")

    combiner.combine.assert_called_once_with(
        {
            "SPY": raw_data,
        }
    )

    cleaner.clean.assert_called_once_with(
        combined_data,
    )

    writer.write.assert_called_once_with(
        cleaned_data,
    )

    assert result == output_path


def test_process_reads_all_tickers() -> None:
    reader = Mock()
    combiner = Mock()
    cleaner = Mock()
    writer = Mock()

    reader.read.return_value = pd.DataFrame()

    combiner.combine.return_value = pd.DataFrame()

    cleaner.clean.return_value = pd.DataFrame()

    writer.write.return_value = Path("market_data.csv")

    service = MarketDataProcessingService(
        reader=reader,
        combiner=combiner,
        cleaner=cleaner,
        writer=writer,
    )

    service.process(
        [
            "SPY",
            "QQQ",
            "IWM",
        ]
    )

    assert reader.read.call_count == 3

    reader.read.assert_any_call("SPY")
    reader.read.assert_any_call("QQQ")
    reader.read.assert_any_call("IWM")

from pathlib import Path

import pandas as pd

from backend.src.services.market_feature_processing_service import (
    MarketFeatureProcessingService,
)
from backend.src.storage.interim_market_data_reader import (
    InterimMarketDataReader,
)
from backend.src.storage.processed_market_data_writer import (
    ProcessedMarketDataWriter,
)


def test_market_feature_processing_service_creates_processed_data(
    tmp_path: Path,
) -> None:
    interim_directory = tmp_path / "interim"
    processed_directory = tmp_path / "processed"

    interim_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    spy_dates = pd.date_range(
        start="2025-01-01",
        periods=25,
    )

    qqq_dates = pd.date_range(
        start="2025-01-01",
        periods=25,
    )

    spy_data = pd.DataFrame(
        {
            "date": spy_dates,
            "ticker": ["SPY"] * 25,
            "open": [100.0 + index for index in range(25)],
            "high": [102.0 + index for index in range(25)],
            "low": [99.0 + index for index in range(25)],
            "close": [101.0 + index for index in range(25)],
            "adjusted_close": [101.0 + index for index in range(25)],
            "volume": [1000 + index * 10 for index in range(25)],
        }
    )

    qqq_data = pd.DataFrame(
        {
            "date": qqq_dates,
            "ticker": ["QQQ"] * 25,
            "open": [200.0 + index for index in range(25)],
            "high": [202.0 + index for index in range(25)],
            "low": [199.0 + index for index in range(25)],
            "close": [201.0 + index for index in range(25)],
            "adjusted_close": [201.0 + index for index in range(25)],
            "volume": [2000 + index * 20 for index in range(25)],
        }
    )

    input_data = pd.concat(
        [
            spy_data,
            qqq_data,
        ],
        ignore_index=True,
    )

    input_path = interim_directory / "market_data.csv"

    input_data.to_csv(
        input_path,
        index=False,
        date_format="%Y-%m-%d",
    )

    reader = InterimMarketDataReader(
        data_directory=interim_directory,
    )

    writer = ProcessedMarketDataWriter(
        output_directory=processed_directory,
    )

    service = MarketFeatureProcessingService(
        reader=reader,
        writer=writer,
    )

    output_path = service.process()

    assert output_path == (processed_directory / "market_features.csv")
    assert output_path.exists()

    result = pd.read_csv(
        output_path,
        parse_dates=["date"],
    )

    expected_feature_columns = {
        "daily_return",
        "log_return",
        "price_change",
        "volume_change",
        "moving_average_5",
        "moving_average_20",
        "volatility_5",
        "volatility_20",
        "momentum_5",
        "momentum_20",
    }

    assert expected_feature_columns.issubset(result.columns)

    assert not result.empty
    assert not result.isna().any().any()

    assert set(result["ticker"]) == {
        "SPY",
        "QQQ",
    }

    assert result.groupby("ticker").size().to_dict() == {
        "QQQ": 5,
        "SPY": 5,
    }

    expected_order = result.sort_values(
        by=["ticker", "date"],
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        result,
        expected_order,
    )

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.src.pipelines.market_data_pipeline import (
    MarketDataPipeline,
)


def test_run_downloads_and_processes_market_data() -> None:
    historical_data_service = Mock()
    processing_service = Mock()
    feature_processing_service = Mock()

    output_path = Path("data/processed/market/market_features.csv")

    feature_processing_service.process.return_value = output_path

    pipeline = MarketDataPipeline(
        historical_data_service=historical_data_service,
        processing_service=processing_service,
        feature_processing_service=feature_processing_service,
    )

    result = pipeline.run(
        tickers=["SPY", "QQQ"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    historical_data_service.download_many.assert_called_once_with(
        tickers=("SPY", "QQQ"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    processing_service.process.assert_called_once_with(
        ["SPY", "QQQ"],
    )

    feature_processing_service.process.assert_called_once_with()

    assert result == output_path


def test_run_normalizes_tickers() -> None:
    historical_data_service = Mock()
    processing_service = Mock()
    feature_processing_service = Mock()

    feature_processing_service.process.return_value = Path(
        "data/processed/market/market_features.csv"
    )

    pipeline = MarketDataPipeline(
        historical_data_service=historical_data_service,
        processing_service=processing_service,
        feature_processing_service=feature_processing_service,
    )

    pipeline.run(
        tickers=[
            " spy ",
            "QQQ",
            "SPY",
            "",
        ],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    historical_data_service.download_many.assert_called_once_with(
        tickers=("SPY", "QQQ"),
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
    )

    processing_service.process.assert_called_once_with(
        ["SPY", "QQQ"],
    )

    feature_processing_service.process.assert_called_once_with()


def test_run_raises_value_error_for_empty_tickers() -> None:
    historical_data_service = Mock()
    processing_service = Mock()
    feature_processing_service = Mock()

    pipeline = MarketDataPipeline(
        historical_data_service=historical_data_service,
        processing_service=processing_service,
        feature_processing_service=feature_processing_service,
    )

    with pytest.raises(
        ValueError,
        match="Lista tickerów nie może być pusta.",
    ):
        pipeline.run(
            tickers=[],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )

    historical_data_service.download_many.assert_not_called()
    processing_service.process.assert_not_called()
    feature_processing_service.process.assert_not_called()

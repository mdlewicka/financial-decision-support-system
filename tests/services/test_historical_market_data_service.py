from datetime import date
from pathlib import Path
from unittest.mock import Mock, call, patch

import pandas as pd
import pytest

from backend.src.config import SUPPORTED_TICKERS
from backend.src.services.historical_market_data_service import (
    HistoricalMarketDataService,
)
from backend.src.services.market_data_batch_result import (
    MarketDataBatchResult,
)


def test_download_history_runs_complete_pipeline() -> None:
    provider = Mock()
    standardizer = Mock()
    validator = Mock()
    writer = Mock()

    raw_data = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Adj Close": [103.0],
            "Volume": [1000],
        }
    )

    standardized_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02"]),
            "ticker": ["SPY"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "adjusted_close": [103.0],
            "volume": [1000],
        }
    )

    validated_data = standardized_data.copy()
    expected_path = Path("data/raw/market/SPY.csv")

    provider.fetch_history.return_value = raw_data
    standardizer.standardize.return_value = standardized_data
    validator.validate.return_value = validated_data
    writer.save.return_value = expected_path

    service = HistoricalMarketDataService(
        provider=provider,
        standardizer=standardizer,
        validator=validator,
        writer=writer,
    )

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)

    result = service.download_history(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
    )

    assert result == expected_path

    provider.fetch_history.assert_called_once_with(
        ticker="SPY",
        start_date=start_date,
        end_date=end_date,
    )

    standardizer.standardize.assert_called_once_with(
        data=raw_data,
        ticker="SPY",
    )

    validator.validate.assert_called_once_with(
        data=standardized_data,
        expected_ticker="SPY",
    )

    writer.save.assert_called_once_with(
        data=validated_data,
        ticker="SPY",
    )


def test_download_history_stops_when_validation_fails() -> None:
    provider = Mock()
    standardizer = Mock()
    validator = Mock()
    writer = Mock()

    raw_data = pd.DataFrame({"Close": [100.0]})
    standardized_data = pd.DataFrame({"close": [100.0]})

    provider.fetch_history.return_value = raw_data
    standardizer.standardize.return_value = standardized_data
    validator.validate.side_effect = ValueError("Niepoprawne dane.")

    service = HistoricalMarketDataService(
        provider=provider,
        standardizer=standardizer,
        validator=validator,
        writer=writer,
    )

    with pytest.raises(ValueError, match="Niepoprawne dane"):
        service.download_history(
            ticker="SPY",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )

    writer.save.assert_not_called()


def test_download_many_downloads_each_unique_ticker() -> None:
    service = HistoricalMarketDataService()

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)

    spy_path = Path("data/raw/market/SPY.csv")
    bitcoin_path = Path("data/raw/market/BTC-USD.csv")
    qqq_path = Path("data/raw/market/QQQ.csv")

    with patch.object(
        service,
        "download_history",
        side_effect=[spy_path, bitcoin_path, qqq_path],
    ) as download_history:
        result = service.download_many(
            tickers=["spy", "BTC-USD", "SPY", " qqq "],
            start_date=start_date,
            end_date=end_date,
        )

    assert result == {
        "SPY": spy_path,
        "BTC-USD": bitcoin_path,
        "QQQ": qqq_path,
    }

    assert download_history.call_args_list == [
        call(
            ticker="SPY",
            start_date=start_date,
            end_date=end_date,
        ),
        call(
            ticker="BTC-USD",
            start_date=start_date,
            end_date=end_date,
        ),
        call(
            ticker="QQQ",
            start_date=start_date,
            end_date=end_date,
        ),
    ]


def test_download_many_rejects_empty_ticker_list() -> None:
    service = HistoricalMarketDataService()

    with pytest.raises(
        ValueError,
        match="Lista tickerów nie może być pusta",
    ):
        service.download_many(
            tickers=[],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )


def test_download_many_rejects_blank_tickers() -> None:
    service = HistoricalMarketDataService()

    with pytest.raises(
        ValueError,
        match="Lista tickerów nie może być pusta",
    ):
        service.download_many(
            tickers=["", " ", "   "],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
        )


def test_download_supported_history_uses_configured_tickers() -> None:
    service = HistoricalMarketDataService()

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)
    expected_result = {
        ticker: Path(f"data/raw/market/{ticker}.csv") for ticker in SUPPORTED_TICKERS
    }

    with patch.object(
        service,
        "download_many",
        return_value=expected_result,
    ) as download_many:
        result = service.download_supported_history(
            start_date=start_date,
            end_date=end_date,
        )

    assert result == expected_result

    download_many.assert_called_once_with(
        tickers=SUPPORTED_TICKERS,
        start_date=start_date,
        end_date=end_date,
    )


def test_download_many_safe_continues_after_ticker_failure() -> None:
    service = HistoricalMarketDataService()

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)

    spy_path = Path("data/raw/market/SPY.csv")
    qqq_path = Path("data/raw/market/QQQ.csv")

    with patch.object(
        service,
        "download_history",
        side_effect=[
            spy_path,
            RuntimeError("Brak danych dla instrumentu."),
            qqq_path,
        ],
    ) as download_history:
        result = service.download_many_safe(
            tickers=["SPY", "INVALID", "QQQ"],
            start_date=start_date,
            end_date=end_date,
        )

    assert isinstance(result, MarketDataBatchResult)

    assert result.successful == {
        "SPY": spy_path,
        "QQQ": qqq_path,
    }

    assert set(result.failed) == {"INVALID"}

    failure = result.failed["INVALID"]

    assert failure.ticker == "INVALID"
    assert failure.error_type == "RuntimeError"
    assert failure.message == "Brak danych dla instrumentu."

    assert result.is_successful is False
    assert result.total_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1

    assert download_history.call_count == 3


def test_download_many_safe_reports_full_success() -> None:
    service = HistoricalMarketDataService()

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)

    spy_path = Path("data/raw/market/SPY.csv")
    qqq_path = Path("data/raw/market/QQQ.csv")

    with patch.object(
        service,
        "download_history",
        side_effect=[spy_path, qqq_path],
    ):
        result = service.download_many_safe(
            tickers=["SPY", "QQQ"],
            start_date=start_date,
            end_date=end_date,
        )

    assert result.successful == {
        "SPY": spy_path,
        "QQQ": qqq_path,
    }
    assert result.failed == {}
    assert result.is_successful is True
    assert result.total_count == 2
    assert result.success_count == 2
    assert result.failure_count == 0


def test_download_supported_history_safe_uses_configured_tickers() -> None:
    service = HistoricalMarketDataService()

    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 10)
    expected_result = MarketDataBatchResult()

    with patch.object(
        service,
        "download_many_safe",
        return_value=expected_result,
    ) as download_many_safe:
        result = service.download_supported_history_safe(
            start_date=start_date,
            end_date=end_date,
        )

    assert result == expected_result

    download_many_safe.assert_called_once_with(
        tickers=SUPPORTED_TICKERS,
        start_date=start_date,
        end_date=end_date,
    )

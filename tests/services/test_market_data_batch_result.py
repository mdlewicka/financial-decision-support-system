from pathlib import Path

from backend.src.services.market_data_batch_result import (
    MarketDataBatchResult,
    MarketDataDownloadFailure,
)


def test_batch_result_calculates_summary() -> None:
    result = MarketDataBatchResult(
        successful={
            "SPY": Path("SPY.csv"),
            "QQQ": Path("QQQ.csv"),
        },
        failed={
            "INVALID": MarketDataDownloadFailure(
                ticker="INVALID",
                error_type="RuntimeError",
                message="Brak danych.",
            )
        },
    )

    assert result.is_successful is False
    assert result.total_count == 3
    assert result.success_count == 2
    assert result.failure_count == 1


def test_empty_batch_result_is_successful() -> None:
    result = MarketDataBatchResult()

    assert result.is_successful is True
    assert result.total_count == 0
    assert result.success_count == 0
    assert result.failure_count == 0

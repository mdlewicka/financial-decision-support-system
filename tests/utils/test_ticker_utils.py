import pytest

from backend.src.utils.ticker_utils import (
    normalize_tickers,
)


def test_normalize_tickers_normalizes_values() -> None:
    result = normalize_tickers(
        [
            " spy ",
            "qqq",
        ]
    )

    assert result == (
        "SPY",
        "QQQ",
    )


def test_normalize_tickers_removes_duplicates() -> None:
    result = normalize_tickers(
        [
            "SPY",
            "spy",
            " SPY ",
            "QQQ",
        ]
    )

    assert result == (
        "SPY",
        "QQQ",
    )


def test_normalize_tickers_ignores_empty_values() -> None:
    result = normalize_tickers(
        [
            "",
            "   ",
            "SPY",
        ]
    )

    assert result == ("SPY",)


def test_normalize_tickers_preserves_order() -> None:
    result = normalize_tickers(
        [
            "QQQ",
            "SPY",
            "IWM",
        ]
    )

    assert result == (
        "QQQ",
        "SPY",
        "IWM",
    )


def test_normalize_tickers_raises_error_for_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match="Lista tickerów nie może być pusta.",
    ):
        normalize_tickers([])


def test_normalize_tickers_raises_error_for_blank_values() -> None:
    with pytest.raises(
        ValueError,
        match="Lista tickerów nie może być pusta.",
    ):
        normalize_tickers(
            [
                "",
                "   ",
            ]
        )

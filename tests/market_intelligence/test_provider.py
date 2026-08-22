from datetime import UTC, date, datetime

import pandas as pd
import pytest

from backend.src.market_intelligence.provider import (
    YahooFinanceMarketIntelligenceProvider,
)

NOW = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)


class FakeHistoryProvider:
    def __init__(self, history: pd.DataFrame) -> None:
        self.history = history
        self.calls: list[dict[str, object]] = []

    def fetch_history(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self.history.copy()


def create_history() -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0]},
        index=pd.to_datetime(["2026-08-11", "2026-08-12", "2026-08-13"]),
    )


def test_fetch_quote_returns_last_two_closes_and_utc_metadata() -> None:
    history_provider = FakeHistoryProvider(create_history())
    provider = YahooFinanceMarketIntelligenceProvider(
        history_provider=history_provider,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    quote = provider.fetch_quote(" spy ")

    assert quote.ticker == "SPY"
    assert quote.price == 101.0
    assert quote.previous_close == 102.0
    assert quote.market_date == date(2026, 8, 13)
    assert quote.retrieved_at_utc == NOW
    assert quote.market_timezone == "America/New_York"
    assert quote.currency == "USD"
    assert quote.source == "Yahoo Finance"
    assert history_provider.calls == [
        {
            "ticker": "SPY",
            "start_date": date(2026, 8, 5),
            "end_date": date(2026, 8, 15),
            "interval": "1d",
        }
    ]


def test_fetch_quote_uses_utc_market_timezone_for_bitcoin() -> None:
    provider = YahooFinanceMarketIntelligenceProvider(
        history_provider=FakeHistoryProvider(create_history()),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    quote = provider.fetch_quote("BTC-USD")

    assert quote.market_timezone == "UTC"


@pytest.mark.parametrize(
    "history, message",
    [
        (
            pd.DataFrame(
                {"Open": [1.0, 2.0]},
                index=pd.to_datetime(["2026-08-12", "2026-08-13"]),
            ),
            "Close column",
        ),
        (
            pd.DataFrame(
                {"Close": [100.0]},
                index=pd.to_datetime(["2026-08-13"]),
            ),
            "fewer than two",
        ),
        (
            pd.DataFrame(
                {"Close": [100.0, -1.0]},
                index=pd.to_datetime(["2026-08-12", "2026-08-13"]),
            ),
            "invalid close",
        ),
    ],
)
def test_fetch_quote_rejects_invalid_history(
    history: pd.DataFrame,
    message: str,
) -> None:
    provider = YahooFinanceMarketIntelligenceProvider(
        history_provider=FakeHistoryProvider(history),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match=message):
        provider.fetch_quote("SPY")


def test_fetch_quote_rejects_unsupported_ticker() -> None:
    provider = YahooFinanceMarketIntelligenceProvider(
        history_provider=FakeHistoryProvider(create_history()),  # type: ignore[arg-type]
        clock=lambda: NOW,
    )

    with pytest.raises(ValueError, match="Unsupported"):
        provider.fetch_quote("AAPL")


def test_provider_rejects_naive_clock() -> None:
    provider = YahooFinanceMarketIntelligenceProvider(
        history_provider=FakeHistoryProvider(create_history()),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 8, 14, 12, 30),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        provider.fetch_quote("SPY")

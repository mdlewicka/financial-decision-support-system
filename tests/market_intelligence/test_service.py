from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from backend.src.market_intelligence.models import (
    MarketDataDelivery,
    MarketDataFreshness,
    ProviderMarketQuote,
)
from backend.src.market_intelligence.service import MarketIntelligenceService

NOW = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)


def create_quote(
    ticker: str,
    *,
    price: float = 102.0,
    previous_close: float = 100.0,
    retrieved_at: datetime = NOW,
) -> ProviderMarketQuote:
    return ProviderMarketQuote(
        ticker=ticker,
        price=price,
        previous_close=previous_close,
        market_date=date(2026, 8, 13),
        retrieved_at_utc=retrieved_at,
        market_timezone=("UTC" if ticker == "BTC-USD" else "America/New_York"),
        currency="USD",
    )


class FakeProvider:
    def __init__(
        self,
        quotes: dict[str, ProviderMarketQuote] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.quotes = quotes or {}
        self.failures = failures or {}
        self.calls: list[str] = []

    def fetch_quote(self, ticker: str) -> ProviderMarketQuote:
        self.calls.append(ticker)

        if ticker in self.failures:
            raise self.failures[ticker]

        return self.quotes[ticker]


def create_service(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    now: datetime = NOW,
    ttl: int = 900,
) -> MarketIntelligenceService:
    return MarketIntelligenceService(
        provider=provider,
        cache_path=tmp_path / "market_intelligence.json",
        cache_ttl_seconds=ttl,
        clock=lambda: now,
    )


def test_live_snapshot_calculates_change_and_writes_cache(tmp_path: Path) -> None:
    provider = FakeProvider({"SPY": create_quote("SPY")})
    service = create_service(tmp_path, provider)

    result = service.get_snapshot(["SPY"])

    assert result.generated_at_utc == NOW
    assert result.cache_ttl_seconds == 900
    assert result.refresh_requested is False
    assert result.available_count == 1
    assert result.stale_count == 0
    quote = result.quotes[0]
    assert quote.price == 102.0
    assert quote.previous_close == 100.0
    assert quote.daily_change == 2.0
    assert quote.daily_change_percent == pytest.approx(0.02)
    assert quote.freshness is MarketDataFreshness.FRESH
    assert quote.delivery is MarketDataDelivery.LIVE
    assert quote.age_seconds == 0
    assert quote.error is None
    assert (tmp_path / "market_intelligence.json").is_file()
    assert not (tmp_path / "market_intelligence.tmp").exists()


def test_second_request_uses_fresh_cache_without_provider_call(
    tmp_path: Path,
) -> None:
    initial_provider = FakeProvider({"SPY": create_quote("SPY")})
    create_service(tmp_path, initial_provider).get_snapshot(["SPY"])
    cached_provider = FakeProvider(failures={"SPY": RuntimeError("offline")})
    service = create_service(
        tmp_path,
        cached_provider,
        now=NOW + timedelta(minutes=5),
    )

    result = service.get_snapshot(["SPY"])

    assert cached_provider.calls == []
    assert result.quotes[0].delivery is MarketDataDelivery.CACHE
    assert result.quotes[0].freshness is MarketDataFreshness.FRESH
    assert result.quotes[0].age_seconds == 300


def test_force_refresh_bypasses_fresh_cache(tmp_path: Path) -> None:
    create_service(
        tmp_path,
        FakeProvider({"SPY": create_quote("SPY")}),
    ).get_snapshot(["SPY"])
    refreshed_quote = create_quote(
        "SPY",
        price=105.0,
        previous_close=102.0,
        retrieved_at=NOW + timedelta(minutes=1),
    )
    provider = FakeProvider({"SPY": refreshed_quote})
    service = create_service(
        tmp_path,
        provider,
        now=NOW + timedelta(minutes=1),
    )

    result = service.get_snapshot(["SPY"], force_refresh=True)

    assert provider.calls == ["SPY"]
    assert result.refresh_requested is True
    assert result.quotes[0].price == 105.0
    assert result.quotes[0].delivery is MarketDataDelivery.LIVE


def test_network_failure_returns_stale_offline_cache(tmp_path: Path) -> None:
    create_service(
        tmp_path,
        FakeProvider({"SPY": create_quote("SPY")}),
    ).get_snapshot(["SPY"])
    service = create_service(
        tmp_path,
        FakeProvider(failures={"SPY": ConnectionError("network unavailable")}),
        now=NOW + timedelta(hours=2),
    )

    result = service.get_snapshot(["SPY"])

    quote = result.quotes[0]
    assert quote.price == 102.0
    assert quote.freshness is MarketDataFreshness.STALE
    assert quote.delivery is MarketDataDelivery.OFFLINE_CACHE
    assert quote.age_seconds == 7200
    assert quote.error == "ConnectionError: network unavailable"


def test_network_failure_without_cache_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    service = create_service(
        tmp_path,
        FakeProvider(failures={"SPY": TimeoutError("timeout")}),
    )

    result = service.get_snapshot(["SPY"])

    quote = result.quotes[0]
    assert quote.available is False
    assert quote.price is None
    assert quote.freshness is MarketDataFreshness.UNAVAILABLE
    assert quote.delivery is MarketDataDelivery.UNAVAILABLE
    assert quote.market_timezone == "America/New_York"
    assert quote.source == "Yahoo Finance"
    assert quote.error == "TimeoutError: timeout"


def test_partial_provider_failure_does_not_hide_other_quotes(
    tmp_path: Path,
) -> None:
    provider = FakeProvider(
        quotes={"SPY": create_quote("SPY")},
        failures={"BTC-USD": ConnectionError("offline")},
    )
    service = create_service(tmp_path, provider)

    result = service.get_snapshot(["SPY", "BTC-USD"])

    assert result.available_count == 1
    assert [quote.ticker for quote in result.quotes] == ["SPY", "BTC-USD"]
    assert result.quotes[0].delivery is MarketDataDelivery.LIVE
    assert result.quotes[1].delivery is MarketDataDelivery.UNAVAILABLE
    assert result.quotes[1].market_timezone == "UTC"


def test_corrupted_cache_is_ignored(tmp_path: Path) -> None:
    cache_path = tmp_path / "market_intelligence.json"
    cache_path.write_text("not-json", encoding="utf-8")
    provider = FakeProvider({"SPY": create_quote("SPY")})
    service = create_service(tmp_path, provider)

    result = service.get_snapshot(["SPY"])

    assert provider.calls == ["SPY"]
    assert result.quotes[0].delivery is MarketDataDelivery.LIVE


def test_cache_with_unexpected_json_shape_is_ignored(tmp_path: Path) -> None:
    cache_path = tmp_path / "market_intelligence.json"
    cache_path.write_text("[]", encoding="utf-8")
    provider = FakeProvider({"SPY": create_quote("SPY")})
    service = create_service(tmp_path, provider)

    result = service.get_snapshot(["SPY"])

    assert provider.calls == ["SPY"]
    assert result.quotes[0].delivery is MarketDataDelivery.LIVE


def test_service_rejects_unsupported_ticker(tmp_path: Path) -> None:
    service = create_service(tmp_path, FakeProvider())

    with pytest.raises(ValueError, match="AAPL"):
        service.get_snapshot(["AAPL"])


def test_service_rejects_invalid_provider_quote(tmp_path: Path) -> None:
    invalid_quote = create_quote("SPY", price=float("nan"))
    service = create_service(
        tmp_path,
        FakeProvider({"SPY": invalid_quote}),
    )

    result = service.get_snapshot(["SPY"])

    assert result.quotes[0].freshness is MarketDataFreshness.UNAVAILABLE
    assert "invalid prices" in str(result.quotes[0].error)


def test_service_rejects_naive_clock(tmp_path: Path) -> None:
    service = MarketIntelligenceService(
        provider=FakeProvider(),
        cache_path=tmp_path / "cache.json",
        clock=lambda: datetime(2026, 8, 14, 12, 30),
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        service.get_snapshot(["SPY"])

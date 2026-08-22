from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from backend.src.api.routes.market_intelligence import (
    get_market_intelligence_service,
)
from backend.src.config import SUPPORTED_TICKERS
from backend.src.main import create_app
from backend.src.market_intelligence.models import (
    MarketDataDelivery,
    MarketDataFreshness,
    MarketIntelligenceQuote,
    MarketIntelligenceSnapshot,
)

NOW = datetime(2026, 8, 14, 12, 30, tzinfo=UTC)


def create_quote(
    ticker: str,
    *,
    freshness: MarketDataFreshness = MarketDataFreshness.FRESH,
    delivery: MarketDataDelivery = MarketDataDelivery.LIVE,
    available: bool = True,
) -> MarketIntelligenceQuote:
    effective_freshness = freshness if available else MarketDataFreshness.UNAVAILABLE
    effective_delivery = delivery if available else MarketDataDelivery.UNAVAILABLE

    return MarketIntelligenceQuote(
        ticker=ticker,
        price=102.0 if available else None,
        previous_close=100.0 if available else None,
        daily_change=2.0 if available else None,
        daily_change_percent=0.02 if available else None,
        market_date=date(2026, 8, 13) if available else None,
        retrieved_at_utc=NOW if available else None,
        market_timezone=("UTC" if ticker == "BTC-USD" else "America/New_York"),
        currency="USD",
        source="Yahoo Finance",
        freshness=effective_freshness,
        delivery=effective_delivery,
        age_seconds=0 if available else None,
        error=None if available else "ConnectionError: offline",
    )


class StubMarketIntelligenceService:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    def get_snapshot(
        self,
        tickers: Iterable[str],
        *,
        force_refresh: bool = False,
    ) -> MarketIntelligenceSnapshot:
        normalized = tuple(ticker.strip().upper() for ticker in tickers)
        self.calls.append((normalized, force_refresh))

        unsupported = set(normalized).difference(SUPPORTED_TICKERS)
        if unsupported:
            raise ValueError("Unsupported market intelligence tickers: AAPL")

        quotes = tuple(
            create_quote(ticker, available=ticker != "AGG") for ticker in normalized
        )
        return MarketIntelligenceSnapshot(
            generated_at_utc=NOW,
            cache_ttl_seconds=900,
            refresh_requested=force_refresh,
            quotes=quotes,
        )


@pytest.fixture
def market_client() -> Iterator[tuple[TestClient, StubMarketIntelligenceService]]:
    application = create_app()
    service = StubMarketIntelligenceService()
    application.dependency_overrides[get_market_intelligence_service] = lambda: service

    with TestClient(application) as client:
        yield client, service


def test_default_request_returns_supported_universe_and_metadata(
    market_client: tuple[TestClient, StubMarketIntelligenceService],
) -> None:
    client, service = market_client

    response = client.get("/api/v1/market-intelligence/quotes")

    assert response.status_code == 200
    payload = response.json()
    assert service.calls == [(SUPPORTED_TICKERS, False)]
    assert payload["generated_at_utc"] == "2026-08-14T12:30:00Z"
    assert payload["cache_ttl_seconds"] == 900
    assert payload["refresh_requested"] is False
    assert payload["real_time"] is False
    assert payload["available_count"] == len(SUPPORTED_TICKERS) - 1
    assert payload["stale_count"] == 0
    assert "nie są notowaniami czasu rzeczywistego" in payload["disclaimer"]
    assert [quote["ticker"] for quote in payload["quotes"]] == list(SUPPORTED_TICKERS)
    assert payload["quotes"][0]["source"] == "Yahoo Finance"


def test_request_accepts_selected_tickers_and_manual_refresh(
    market_client: tuple[TestClient, StubMarketIntelligenceService],
) -> None:
    client, service = market_client

    response = client.get(
        "/api/v1/market-intelligence/quotes",
        params=[("tickers", "SPY"), ("tickers", "BTC-USD"), ("refresh", "true")],
    )

    assert response.status_code == 200
    assert service.calls == [(("SPY", "BTC-USD"), True)]
    payload = response.json()
    assert payload["refresh_requested"] is True
    assert [quote["market_timezone"] for quote in payload["quotes"]] == [
        "America/New_York",
        "UTC",
    ]


def test_unavailable_quote_is_returned_without_failing_entire_response(
    market_client: tuple[TestClient, StubMarketIntelligenceService],
) -> None:
    client, _ = market_client

    response = client.get(
        "/api/v1/market-intelligence/quotes",
        params={"tickers": "AGG"},
    )

    assert response.status_code == 200
    quote = response.json()["quotes"][0]
    assert quote["price"] is None
    assert quote["freshness"] == "unavailable"
    assert quote["delivery"] == "unavailable"
    assert quote["error"] == "ConnectionError: offline"


def test_unsupported_ticker_returns_422(
    market_client: tuple[TestClient, StubMarketIntelligenceService],
) -> None:
    client, _ = market_client

    response = client.get(
        "/api/v1/market-intelligence/quotes",
        params={"tickers": "AAPL"},
    )

    assert response.status_code == 422
    assert "AAPL" in response.json()["detail"]


def test_market_intelligence_route_is_in_openapi_contract(
    market_client: tuple[TestClient, StubMarketIntelligenceService],
) -> None:
    client, _ = market_client

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/market-intelligence/quotes" in response.json()["paths"]

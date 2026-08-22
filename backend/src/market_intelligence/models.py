"""Typed contracts for current market information."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


class MarketDataFreshness(StrEnum):
    """Age-based freshness classification."""

    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class MarketDataDelivery(StrEnum):
    """How a quote reached the caller."""

    LIVE = "live"
    CACHE = "cache"
    OFFLINE_CACHE = "offline_cache"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ProviderMarketQuote:
    """Validated quote returned by a current-price provider."""

    ticker: str
    price: float
    previous_close: float
    market_date: date
    retrieved_at_utc: datetime
    market_timezone: str
    currency: str
    source: str = "Yahoo Finance"


@dataclass(frozen=True, slots=True)
class MarketIntelligenceQuote:
    """One current quote enriched with freshness and delivery metadata."""

    ticker: str
    price: float | None
    previous_close: float | None
    daily_change: float | None
    daily_change_percent: float | None
    market_date: date | None
    retrieved_at_utc: datetime | None
    market_timezone: str
    currency: str
    source: str
    freshness: MarketDataFreshness
    delivery: MarketDataDelivery
    age_seconds: int | None
    error: str | None = None

    @property
    def available(self) -> bool:
        """Return whether numerical quote data is available."""
        return self.price is not None


@dataclass(frozen=True, slots=True)
class MarketIntelligenceSnapshot:
    """Current-price response for a requested set of instruments."""

    generated_at_utc: datetime
    cache_ttl_seconds: int
    refresh_requested: bool
    quotes: tuple[MarketIntelligenceQuote, ...]

    @property
    def available_count(self) -> int:
        """Return the number of available quotes."""
        return sum(quote.available for quote in self.quotes)

    @property
    def stale_count(self) -> int:
        """Return the number of stale cached quotes."""
        return sum(
            quote.freshness is MarketDataFreshness.STALE for quote in self.quotes
        )

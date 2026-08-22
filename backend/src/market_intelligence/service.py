"""Cache-aware current market information with an offline fallback."""

import json
import math
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol, cast

from backend.src.config import DATA_DIR, SUPPORTED_ASSETS, SUPPORTED_TICKERS
from backend.src.market_intelligence.models import (
    MarketDataDelivery,
    MarketDataFreshness,
    MarketIntelligenceQuote,
    MarketIntelligenceSnapshot,
    ProviderMarketQuote,
)
from backend.src.market_intelligence.provider import (
    YahooFinanceMarketIntelligenceProvider,
)
from backend.src.utils.ticker_utils import normalize_tickers

_UTC = UTC
_CACHE_SCHEMA_VERSION = 1
DEFAULT_MARKET_INTELLIGENCE_CACHE_PATH = DATA_DIR / "cache" / "market_intelligence.json"


class CurrentMarketQuoteProvider(Protocol):
    """Provider boundary used by the service and deterministic tests."""

    def fetch_quote(self, ticker: str) -> ProviderMarketQuote:
        """Fetch one current market quote."""
        ...


class MarketIntelligenceService:
    """Return current quotes without failing when the network is unavailable."""

    def __init__(
        self,
        provider: CurrentMarketQuoteProvider | None = None,
        *,
        cache_path: Path = DEFAULT_MARKET_INTELLIGENCE_CACHE_PATH,
        cache_ttl_seconds: int = 900,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive.")

        self._provider = provider or YahooFinanceMarketIntelligenceProvider()
        self._cache_path = cache_path
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock or (lambda: datetime.now(_UTC))

    def get_snapshot(
        self,
        tickers: Iterable[str] = SUPPORTED_TICKERS,
        *,
        force_refresh: bool = False,
    ) -> MarketIntelligenceSnapshot:
        """Return requested quotes, using cache and offline fallback as needed."""
        requested_tickers = normalize_tickers(tickers)
        self._validate_supported_tickers(requested_tickers)
        generated_at = self._now_utc()
        cached_quotes = self._read_cache()
        output: list[MarketIntelligenceQuote] = []
        updated_cache = dict(cached_quotes)

        for ticker in requested_tickers:
            cached_quote = cached_quotes.get(ticker)
            cached_age = self._age_seconds(
                cached_quote,
                generated_at,
            )

            if (
                not force_refresh
                and cached_quote is not None
                and cached_age is not None
                and cached_age <= self._cache_ttl_seconds
            ):
                output.append(
                    self._to_market_quote(
                        cached_quote,
                        generated_at=generated_at,
                        delivery=MarketDataDelivery.CACHE,
                    )
                )
                continue

            try:
                live_quote = self._provider.fetch_quote(ticker)
                self._validate_provider_quote(live_quote, expected_ticker=ticker)
            except Exception as error:
                output.append(
                    self._offline_or_unavailable_quote(
                        ticker=ticker,
                        cached_quote=cached_quote,
                        generated_at=generated_at,
                        error=error,
                    )
                )
            else:
                updated_cache[ticker] = live_quote
                output.append(
                    self._to_market_quote(
                        live_quote,
                        generated_at=generated_at,
                        delivery=MarketDataDelivery.LIVE,
                    )
                )

        if updated_cache != cached_quotes:
            self._write_cache(updated_cache)

        return MarketIntelligenceSnapshot(
            generated_at_utc=generated_at,
            cache_ttl_seconds=self._cache_ttl_seconds,
            refresh_requested=force_refresh,
            quotes=tuple(output),
        )

    def _offline_or_unavailable_quote(
        self,
        *,
        ticker: str,
        cached_quote: ProviderMarketQuote | None,
        generated_at: datetime,
        error: Exception,
    ) -> MarketIntelligenceQuote:
        message = f"{type(error).__name__}: {error}"

        if cached_quote is not None:
            return self._to_market_quote(
                cached_quote,
                generated_at=generated_at,
                delivery=MarketDataDelivery.OFFLINE_CACHE,
                error=message,
            )

        asset = next(
            asset for asset in SUPPORTED_ASSETS.values() if asset.ticker == ticker
        )

        return MarketIntelligenceQuote(
            ticker=ticker,
            price=None,
            previous_close=None,
            daily_change=None,
            daily_change_percent=None,
            market_date=None,
            retrieved_at_utc=None,
            market_timezone=("UTC" if ticker == "BTC-USD" else "America/New_York"),
            currency=asset.quote_currency,
            source="Yahoo Finance",
            freshness=MarketDataFreshness.UNAVAILABLE,
            delivery=MarketDataDelivery.UNAVAILABLE,
            age_seconds=None,
            error=message,
        )

    def _to_market_quote(
        self,
        quote: ProviderMarketQuote,
        *,
        generated_at: datetime,
        delivery: MarketDataDelivery,
        error: str | None = None,
    ) -> MarketIntelligenceQuote:
        age_seconds = self._age_seconds(quote, generated_at)
        assert age_seconds is not None
        freshness = (
            MarketDataFreshness.FRESH
            if age_seconds <= self._cache_ttl_seconds
            else MarketDataFreshness.STALE
        )
        daily_change = quote.price - quote.previous_close
        daily_change_percent = daily_change / quote.previous_close

        return MarketIntelligenceQuote(
            ticker=quote.ticker,
            price=quote.price,
            previous_close=quote.previous_close,
            daily_change=daily_change,
            daily_change_percent=daily_change_percent,
            market_date=quote.market_date,
            retrieved_at_utc=quote.retrieved_at_utc,
            market_timezone=quote.market_timezone,
            currency=quote.currency,
            source=quote.source,
            freshness=freshness,
            delivery=delivery,
            age_seconds=age_seconds,
            error=error,
        )

    def _read_cache(self) -> dict[str, ProviderMarketQuote]:
        if not self._cache_path.exists():
            return {}

        try:
            payload = json.loads(self._cache_path.read_text(encoding="utf-8"))

            if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
                return {}

            return {
                item["ticker"]: self._deserialize_quote(item)
                for item in payload.get("quotes", [])
            }
        except (
            AttributeError,
            OSError,
            TypeError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ):
            return {}

    def _write_cache(self, quotes: dict[str, ProviderMarketQuote]) -> None:
        payload = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "quotes": [
                self._serialize_quote(quotes[ticker]) for ticker in sorted(quotes)
            ],
        }
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._cache_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._cache_path)

    @staticmethod
    def _serialize_quote(quote: ProviderMarketQuote) -> dict[str, object]:
        return {
            "ticker": quote.ticker,
            "price": quote.price,
            "previous_close": quote.previous_close,
            "market_date": quote.market_date.isoformat(),
            "retrieved_at_utc": quote.retrieved_at_utc.isoformat(),
            "market_timezone": quote.market_timezone,
            "currency": quote.currency,
            "source": quote.source,
        }

    @staticmethod
    def _deserialize_quote(payload: dict[str, object]) -> ProviderMarketQuote:
        retrieved_at = datetime.fromisoformat(str(payload["retrieved_at_utc"]))

        if retrieved_at.tzinfo is None:
            raise ValueError("Cached retrieved_at_utc must be timezone-aware.")

        price = cast(float | int, payload["price"])
        previous_close = cast(float | int, payload["previous_close"])

        quote = ProviderMarketQuote(
            ticker=str(payload["ticker"]),
            price=float(price),
            previous_close=float(previous_close),
            market_date=date.fromisoformat(str(payload["market_date"])),
            retrieved_at_utc=retrieved_at.astimezone(_UTC),
            market_timezone=str(payload["market_timezone"]),
            currency=str(payload["currency"]),
            source=str(payload["source"]),
        )

        if quote.ticker not in SUPPORTED_TICKERS:
            raise ValueError("Cached ticker is not supported.")

        if not all(
            math.isfinite(value) and value > 0
            for value in (quote.price, quote.previous_close)
        ):
            raise ValueError("Cached prices are invalid.")

        expected_timezone = "UTC" if quote.ticker == "BTC-USD" else "America/New_York"

        if quote.market_timezone != expected_timezone:
            raise ValueError("Cached market timezone is invalid.")

        if quote.currency != "USD" or quote.source != "Yahoo Finance":
            raise ValueError("Cached source metadata is invalid.")

        return quote

    @staticmethod
    def _age_seconds(
        quote: ProviderMarketQuote | None,
        generated_at: datetime,
    ) -> int | None:
        if quote is None:
            return None

        return max(
            0,
            int((generated_at - quote.retrieved_at_utc).total_seconds()),
        )

    @staticmethod
    def _validate_provider_quote(
        quote: ProviderMarketQuote,
        *,
        expected_ticker: str,
    ) -> None:
        if quote.ticker != expected_ticker:
            raise ValueError("The provider returned a quote for a different ticker.")

        if quote.retrieved_at_utc.tzinfo is None:
            raise ValueError("Provider timestamps must be timezone-aware.")

        if not all(
            math.isfinite(value) and value > 0
            for value in (quote.price, quote.previous_close)
        ):
            raise ValueError("The provider returned invalid prices.")

    @staticmethod
    def _validate_supported_tickers(tickers: tuple[str, ...]) -> None:
        unsupported = sorted(set(tickers).difference(SUPPORTED_TICKERS))

        if unsupported:
            raise ValueError(
                "Unsupported market intelligence tickers: " + ", ".join(unsupported)
            )

    def _now_utc(self) -> datetime:
        value = self._clock()

        if value.tzinfo is None:
            raise ValueError("The market intelligence clock must be timezone-aware.")

        return value.astimezone(_UTC)

"""Yahoo Finance adapter for small-scale current market intelligence."""

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pandas as pd

from backend.src.config import SUPPORTED_ASSETS
from backend.src.market_intelligence.models import ProviderMarketQuote
from backend.src.providers.yahoo_finance import YahooFinanceProvider

_UTC = UTC
_DEFAULT_MARKET_TIMEZONE = "America/New_York"
_CRYPTOCURRENCY_MARKET_TIMEZONE = "UTC"


class YahooFinanceMarketIntelligenceProvider:
    """Read the last two daily closes from the existing Yahoo adapter."""

    def __init__(
        self,
        history_provider: YahooFinanceProvider | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        lookback_days: int = 10,
    ) -> None:
        if lookback_days < 3:
            raise ValueError("lookback_days must be at least 3.")

        self._history_provider = history_provider or YahooFinanceProvider()
        self._clock = clock or (lambda: datetime.now(_UTC))
        self._lookback_days = lookback_days

    def fetch_quote(self, ticker: str) -> ProviderMarketQuote:
        """Fetch the last price and its preceding daily close."""
        normalized_ticker = ticker.strip().upper()
        asset = next(
            (
                candidate
                for candidate in SUPPORTED_ASSETS.values()
                if candidate.ticker == normalized_ticker
            ),
            None,
        )

        if asset is None:
            raise ValueError(f"Unsupported market intelligence ticker: {ticker!r}.")

        retrieved_at = self._normalize_utc(self._clock())
        end_date = retrieved_at.date() + timedelta(days=1)
        start_date = end_date - timedelta(days=self._lookback_days)
        history = self._history_provider.fetch_history(
            ticker=normalized_ticker,
            start_date=start_date,
            end_date=end_date,
            interval="1d",
        )
        close = self._extract_close(history)

        if len(close) < 2:
            raise ValueError(
                f"Yahoo Finance returned fewer than two closes for {normalized_ticker}."
            )

        previous_close = float(close.iloc[-2])
        price = float(close.iloc[-1])

        if not all(
            math.isfinite(value) and value > 0 for value in (price, previous_close)
        ):
            raise ValueError(
                f"Yahoo Finance returned an invalid close for {normalized_ticker}."
            )

        market_date = pd.Timestamp(close.index[-1]).date()
        market_timezone = (
            _CRYPTOCURRENCY_MARKET_TIMEZONE
            if normalized_ticker == "BTC-USD"
            else _DEFAULT_MARKET_TIMEZONE
        )

        return ProviderMarketQuote(
            ticker=normalized_ticker,
            price=price,
            previous_close=previous_close,
            market_date=market_date,
            retrieved_at_utc=retrieved_at,
            market_timezone=market_timezone,
            currency=asset.quote_currency,
        )

    @staticmethod
    def _extract_close(history: pd.DataFrame) -> pd.Series:
        if history.empty:
            raise ValueError("Yahoo Finance returned an empty history.")

        close_column = next(
            (
                column
                for column in history.columns
                if str(column).strip().lower() == "close"
            ),
            None,
        )

        if close_column is None:
            raise ValueError("Yahoo Finance history does not contain a Close column.")

        close_values = history[close_column]

        if not isinstance(close_values, pd.Series):
            raise ValueError(
                "Yahoo Finance history contains an ambiguous Close column."
            )

        close = pd.to_numeric(
            close_values,
            errors="coerce",
        ).dropna()

        if close.empty:
            raise ValueError("Yahoo Finance history does not contain valid closes.")

        return close.sort_index()

    @staticmethod
    def _normalize_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("The market intelligence clock must be timezone-aware.")

        return value.astimezone(_UTC)

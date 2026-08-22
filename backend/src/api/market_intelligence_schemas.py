"""HTTP schemas for small-scale current market information."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MarketIntelligenceQuoteResponse(BaseModel):
    """One quote with explicit source, timestamp and freshness metadata."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )

    ticker: str = Field(min_length=1)
    price: float | None
    previous_close: float | None
    daily_change: float | None
    daily_change_percent: float | None
    market_date: date | None
    retrieved_at_utc: datetime | None
    market_timezone: str = Field(min_length=1)
    currency: str = Field(min_length=1)
    source: Literal["Yahoo Finance"]
    freshness: Literal["fresh", "stale", "unavailable"]
    delivery: Literal["live", "cache", "offline_cache", "unavailable"]
    age_seconds: int | None = Field(default=None, ge=0)
    error: str | None = None


class MarketIntelligenceResponse(BaseModel):
    """Current market snapshot returned by the API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at_utc: datetime
    cache_ttl_seconds: int = Field(gt=0)
    refresh_requested: bool
    available_count: int = Field(ge=0)
    stale_count: int = Field(ge=0)
    real_time: Literal[False] = False
    disclaimer: str = Field(min_length=1)
    quotes: tuple[MarketIntelligenceQuoteResponse, ...] = Field(min_length=1)

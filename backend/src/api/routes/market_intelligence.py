"""Current market information API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.src.api.config import API_V1_PREFIX
from backend.src.api.market_intelligence_schemas import (
    MarketIntelligenceQuoteResponse,
    MarketIntelligenceResponse,
)
from backend.src.config import SUPPORTED_TICKERS
from backend.src.market_intelligence.service import MarketIntelligenceService

router = APIRouter(
    prefix=f"{API_V1_PREFIX}/market-intelligence",
    tags=["market-intelligence"],
)

_DISCLAIMER = (
    "Dane mają charakter informacyjny, nie są notowaniami czasu rzeczywistego "
    "ani rekomendacją inwestycyjną."
)


def get_market_intelligence_service() -> MarketIntelligenceService:
    """Create the cache-aware current-price service."""
    return MarketIntelligenceService()


MarketIntelligenceServiceDependency = Annotated[
    MarketIntelligenceService,
    Depends(get_market_intelligence_service),
]
TickerQuery = Annotated[list[str] | None, Query()]
RefreshQuery = Annotated[bool, Query()]


@router.get(
    "/quotes",
    response_model=MarketIntelligenceResponse,
    summary="Get current market prices with freshness metadata",
)
def get_market_intelligence_quotes(
    service: MarketIntelligenceServiceDependency,
    tickers: TickerQuery = None,
    refresh: RefreshQuery = False,
) -> MarketIntelligenceResponse:
    """Return supported quotes and use stale cache when Yahoo is unavailable."""
    try:
        snapshot = service.get_snapshot(
            tickers=tickers or SUPPORTED_TICKERS,
            force_refresh=refresh,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    return MarketIntelligenceResponse(
        generated_at_utc=snapshot.generated_at_utc,
        cache_ttl_seconds=snapshot.cache_ttl_seconds,
        refresh_requested=snapshot.refresh_requested,
        available_count=snapshot.available_count,
        stale_count=snapshot.stale_count,
        disclaimer=_DISCLAIMER,
        quotes=tuple(
            MarketIntelligenceQuoteResponse(
                ticker=quote.ticker,
                price=quote.price,
                previous_close=quote.previous_close,
                daily_change=quote.daily_change,
                daily_change_percent=quote.daily_change_percent,
                market_date=quote.market_date,
                retrieved_at_utc=quote.retrieved_at_utc,
                market_timezone=quote.market_timezone,
                currency=quote.currency,
                source="Yahoo Finance",
                freshness=quote.freshness.value,
                delivery=quote.delivery.value,
                age_seconds=quote.age_seconds,
                error=quote.error,
            )
            for quote in snapshot.quotes
        ),
    )

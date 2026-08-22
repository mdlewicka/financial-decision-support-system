from typing import Any

import pytest

from frontend.api_client import APIClient
from frontend.market_intelligence_view import (
    MARKET_INTELLIGENCE_PATH,
    _build_quotes_frame,
    _format_age,
    _load_market_intelligence,
    _snapshot_status,
)


class RecordingClient(APIClient):
    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__("http://test-api")
        self.response = response
        self.calls: list[tuple[str, float | None]] = []

    def get(
        self,
        path: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self.calls.append((path, timeout_seconds))
        return self.response


def quote(
    ticker: str,
    *,
    freshness: str = "fresh",
    delivery: str = "live",
    price: float | None = 102.0,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "price": price,
        "previous_close": 100.0 if price is not None else None,
        "daily_change": 2.0 if price is not None else None,
        "daily_change_percent": 0.02 if price is not None else None,
        "market_date": "2026-08-13" if price is not None else None,
        "retrieved_at_utc": ("2026-08-14T12:30:00Z" if price is not None else None),
        "market_timezone": "UTC" if ticker == "BTC-USD" else "America/New_York",
        "currency": "USD",
        "source": "Yahoo Finance",
        "freshness": freshness,
        "delivery": delivery,
        "age_seconds": 120 if price is not None else None,
        "error": None,
    }


def snapshot(*quotes: dict[str, Any]) -> dict[str, Any]:
    return {
        "generated_at_utc": "2026-08-14T12:30:00Z",
        "cache_ttl_seconds": 900,
        "refresh_requested": False,
        "available_count": sum(item["price"] is not None for item in quotes),
        "stale_count": sum(item["freshness"] == "stale" for item in quotes),
        "real_time": False,
        "disclaimer": "Informational only.",
        "quotes": list(quotes),
    }


def test_load_market_intelligence_uses_normal_and_refresh_paths() -> None:
    response = snapshot(quote("SPY"))
    client = RecordingClient(response)

    assert _load_market_intelligence(client) == response
    assert _load_market_intelligence(client, force_refresh=True) == response
    assert client.calls == [
        (MARKET_INTELLIGENCE_PATH, None),
        (f"{MARKET_INTELLIGENCE_PATH}?refresh=true", 45.0),
    ]


def test_quotes_frame_exposes_price_change_source_and_freshness() -> None:
    frame = _build_quotes_frame(
        snapshot(
            quote("SPY"),
            quote(
                "BTC-USD",
                freshness="stale",
                delivery="offline_cache",
            ),
        )
    )

    assert list(frame["Ticker"]) == ["SPY", "BTC-USD"]
    assert frame.iloc[0]["Cena"] == 102.0
    assert frame.iloc[0]["Zmiana dzienna"] == 2.0
    assert frame.iloc[0]["Zmiana dzienna %"] == pytest.approx(2.0)
    assert frame.iloc[0]["Świeżość"] == "Świeże"
    assert frame.iloc[1]["Świeżość"] == "Nieaktualne"
    assert frame.iloc[1]["Dostarczenie"] == "Cache offline"
    assert frame.iloc[1]["Strefa rynku"] == "UTC"
    assert frame.iloc[1]["Czas prezentacji"] == "2026-08-14 14:30:00 CEST"


def test_quotes_frame_translates_columns_and_converts_timezone() -> None:
    frame = _build_quotes_frame(
        snapshot(quote("SPY")),
        language="en",
        presentation_timezone="America/New_York",
    )

    assert list(frame.columns) == [
        "Ticker",
        "Price",
        "Currency",
        "Daily change",
        "Daily change %",
        "Market date",
        "Freshness",
        "Data age",
        "Delivery",
        "Presented time",
        "Market time zone",
    ]
    assert frame.iloc[0]["Freshness"] == "Fresh"
    assert frame.iloc[0]["Presented time"] == "2026-08-14 08:30:00 EDT"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, "Brak"),
        (30, "30 s"),
        (120, "2 min"),
        (3720, "1 h 2 min"),
        ("invalid", "Brak"),
    ],
)
def test_format_age_is_readable(seconds: object, expected: str) -> None:
    assert _format_age(seconds) == expected


def test_snapshot_status_distinguishes_complete_partial_and_unavailable() -> None:
    assert _snapshot_status(snapshot(quote("SPY")))[0] == "success"
    assert (
        _snapshot_status(
            snapshot(
                quote("SPY", freshness="stale", delivery="offline_cache"),
                quote(
                    "AGG",
                    freshness="unavailable",
                    delivery="unavailable",
                    price=None,
                ),
            )
        )[0]
        == "warning"
    )
    assert (
        _snapshot_status(
            snapshot(
                quote(
                    "AGG",
                    freshness="unavailable",
                    delivery="unavailable",
                    price=None,
                )
            )
        )[0]
        == "error"
    )
    assert _snapshot_status(snapshot(quote("SPY")), "en") == (
        "success",
        "All prices are available and within the configured cache limit.",
    )


def test_quotes_frame_ignores_invalid_entries() -> None:
    frame = _build_quotes_frame({"quotes": ["invalid", quote("SPY")]})

    assert list(frame["Ticker"]) == ["SPY"]

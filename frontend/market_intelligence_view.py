"""Small current-market information view with explicit freshness metadata."""

from typing import Any, Final, Literal, cast

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient, APIClientError
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import (
    DEFAULT_PRESENTATION_TIMEZONE,
    format_timestamp,
    render_centered_dataframe,
)

MARKET_INTELLIGENCE_PATH: Final = "/api/v1/market-intelligence/quotes"

_FRESHNESS_KEYS: Final[dict[str, str]] = {
    "fresh": "market.fresh",
    "stale": "market.stale",
    "unavailable": "market.unavailable",
}
_DELIVERY_KEYS: Final[dict[str, str]] = {
    "live": "market.live",
    "cache": "market.cache",
    "offline_cache": "market.offline_cache",
    "unavailable": "market.no_data",
}


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    """Return only object entries from a JSON list."""
    if not isinstance(value, list):
        return []

    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def _load_market_intelligence(
    client: APIClient,
    *,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load all supported quotes and optionally bypass a fresh cache."""
    path = MARKET_INTELLIGENCE_PATH

    if force_refresh:
        path += "?refresh=true"

    return client.get(path, timeout_seconds=45.0 if force_refresh else None)


def _format_age(
    value: Any,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Format quote age without implying real-time precision."""
    if value is None:
        return translate("market.missing", language)

    try:
        seconds = max(0, int(value))
    except (TypeError, ValueError):
        return translate("market.missing", language)

    if seconds < 60:
        return f"{seconds} s"

    if seconds < 3600:
        return f"{seconds // 60} min"

    return f"{seconds // 3600} h {seconds % 3600 // 60} min"


def _percentage_points(value: Any) -> float | None:
    """Convert a decimal return to percentage points for display."""
    if value is None:
        return None

    try:
        return float(value) * 100.0
    except (TypeError, ValueError):
        return None


def _build_quotes_frame(
    snapshot: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
    presentation_timezone: str = DEFAULT_PRESENTATION_TIMEZONE,
) -> pd.DataFrame:
    """Build a readable table while retaining missing-data signals."""
    rows: list[dict[str, Any]] = []

    for quote in _as_mapping_list(snapshot.get("quotes")):
        freshness = str(quote.get("freshness", "unavailable"))
        delivery = str(quote.get("delivery", "unavailable"))
        rows.append(
            {
                translate("market.column.ticker", language): str(
                    quote.get("ticker", "")
                ),
                translate("market.column.price", language): quote.get("price"),
                translate("market.column.currency", language): str(
                    quote.get("currency", "")
                ),
                translate("market.column.change", language): quote.get("daily_change"),
                translate("market.column.change_percent", language): (
                    _percentage_points(quote.get("daily_change_percent"))
                ),
                translate("market.column.market_date", language): quote.get(
                    "market_date"
                )
                or translate("market.missing", language),
                translate("market.column.freshness", language): translate(
                    _FRESHNESS_KEYS.get(freshness, "market.unavailable"),
                    language,
                ),
                translate("market.column.age", language): _format_age(
                    quote.get("age_seconds"), language
                ),
                translate("market.column.delivery", language): translate(
                    _DELIVERY_KEYS.get(delivery, "market.no_data"),
                    language,
                ),
                translate("market.column.presented_time", language): format_timestamp(
                    quote.get("retrieved_at_utc"),
                    presentation_timezone,
                    missing=translate("market.missing", language),
                ),
                translate("market.column.market_timezone", language): str(
                    quote.get("market_timezone", "")
                ),
            }
        )

    return pd.DataFrame(rows)


def _snapshot_status(
    snapshot: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> tuple[Literal["success", "warning", "error"], str]:
    """Explain partial availability instead of hiding degraded operation."""
    quotes = _as_mapping_list(snapshot.get("quotes"))
    available = sum(quote.get("price") is not None for quote in quotes)
    stale = sum(quote.get("freshness") == "stale" for quote in quotes)
    unavailable = len(quotes) - available

    if not quotes or available == 0:
        return (
            "error",
            translate("market.status_error", language),
        )

    if stale or unavailable:
        return (
            "warning",
            translate(
                "market.status_partial",
                language,
                stale=stale,
                unavailable=unavailable,
            ),
        )

    return "success", translate("market.status_success", language)


def _render_status(level: str, message: str) -> None:
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)


def render_market_intelligence_page(
    client: APIClient,
    *,
    language: LanguageCode = DEFAULT_LANGUAGE,
    presentation_timezone: str = DEFAULT_PRESENTATION_TIMEZONE,
) -> None:
    """Render current prices without implying streaming or trading capability."""
    st.title(translate("market.title", language))
    st.caption(translate("market.caption", language))

    force_refresh = st.button(
        translate("market.refresh", language),
        key="market_intelligence_refresh",
        icon=":material/refresh:",
        help=translate("market.refresh_help", language),
    )

    try:
        snapshot = _load_market_intelligence(
            client,
            force_refresh=force_refresh,
        )
    except APIClientError as error:
        st.error(translate("market.load_error", language))
        st.caption(str(error))
        return

    if force_refresh:
        st.info(translate("market.refresh_done", language))

    status_level, status_message = _snapshot_status(snapshot, language)
    _render_status(status_level, status_message)

    metric_columns = st.columns(3)
    metric_columns[0].metric(
        translate("market.metric_available", language),
        str(snapshot.get("available_count", 0)),
    )
    metric_columns[1].metric(
        translate("market.metric_stale", language),
        str(snapshot.get("stale_count", 0)),
    )
    metric_columns[2].metric(
        translate("market.metric_cache", language),
        f"{int(snapshot.get('cache_ttl_seconds', 0)) // 60} min",
    )

    frame = _build_quotes_frame(
        snapshot,
        language,
        presentation_timezone,
    )

    if frame.empty:
        st.info(translate("market.empty", language))
    else:
        render_centered_dataframe(
            frame,
            hide_index=True,
            width="stretch",
            column_config={
                translate("market.column.price", language): (
                    st.column_config.NumberColumn(format="%.4f")
                ),
                translate("market.column.change", language): (
                    st.column_config.NumberColumn(format="%+.4f")
                ),
                translate("market.column.change_percent", language): (
                    st.column_config.NumberColumn(format="%+.2f%%")
                ),
            },
        )

    st.caption(translate("market.disclaimer", language))
    st.caption(translate("market.time_note", language))

    with st.expander(translate("market.freshness_help", language)):
        st.markdown(translate("market.freshness_markdown", language))

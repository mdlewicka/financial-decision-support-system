"""Time-zone-safe presentation helpers; source timestamps remain in UTC."""

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

SUPPORTED_PRESENTATION_TIMEZONES: Final[tuple[str, ...]] = (
    "Europe/Warsaw",
    "America/New_York",
    "UTC",
)
DEFAULT_PRESENTATION_TIMEZONE: Final = "Europe/Warsaw"


DataframeWidth = Literal["stretch", "content"] | int


def _extract_number_format(config: Any) -> str | None:
    """Return a Streamlit number-format string when one is configured."""
    if not isinstance(config, Mapping):
        return None

    type_config = config.get("type_config")
    if isinstance(type_config, Mapping):
        format_value = type_config.get("format")
        if isinstance(format_value, str):
            return format_value

    format_value = config.get("format")
    if isinstance(format_value, str):
        return format_value

    return None


def _build_cell_formatter(format_spec: str) -> Callable[[Any], str]:
    """Build a safe formatter for printf-style Streamlit number formats."""

    def formatter(value: Any) -> str:
        try:
            return format_spec % value
        except (TypeError, ValueError):
            return str(value)

    return formatter


def _table_wrapper_style(width: DataframeWidth) -> str:
    """Translate the existing dataframe width contract to safe inline CSS."""
    if width == "stretch":
        return "width:100%;"

    if width == "content":
        return "width:max-content;max-width:100%;"

    pixel_width = max(int(width), 1)
    return f"width:{pixel_width}px;max-width:100%;"


def render_centered_dataframe(
    frame: pd.DataFrame,
    *,
    width: DataframeWidth = "stretch",
    hide_index: bool | None = None,
    column_config: Mapping[str | int, Any] | None = None,
) -> None:
    """Render a presentation table with centered headers and cell values."""
    formatters: dict[Any, Callable[[Any], str]] = {}

    if column_config is not None:
        for column, config in column_config.items():
            format_spec = _extract_number_format(config)
            if format_spec is not None and column in frame.columns:
                formatters[column] = _build_cell_formatter(format_spec)

    table_html = frame.to_html(
        border=0,
        classes="fdss-table",
        escape=True,
        formatters=formatters or None,
        index=hide_index is not True,
        justify="center",
        na_rep="—",
    )

    wrapper_style = _table_wrapper_style(width)

    st.markdown(
        (f'<div class="fdss-table-wrapper" style="{wrapper_style}">{table_html}</div>'),
        unsafe_allow_html=True,
    )


def validate_presentation_timezone(timezone_name: str) -> str:
    """Accept only the small supported presentation-timezone contract."""
    if timezone_name not in SUPPORTED_PRESENTATION_TIMEZONES:
        raise ValueError(f"Unsupported presentation time zone: {timezone_name!r}.")

    return timezone_name


def format_timestamp(
    value: str | datetime | None,
    timezone_name: str,
    *,
    missing: str,
) -> str:
    """Convert an aware ISO/datetime value without mutating source data."""
    if value is None:
        return missing

    validate_presentation_timezone(timezone_name)

    if isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        if not normalized:
            return missing
        try:
            timestamp = datetime.fromisoformat(normalized)
        except ValueError:
            return missing
    else:
        timestamp = value

    if timestamp.tzinfo is None:
        raise ValueError("Presented timestamps must be timezone-aware.")

    try:
        target_timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"Unknown time zone: {timezone_name!r}.") from error

    return timestamp.astimezone(target_timezone).strftime("%Y-%m-%d %H:%M:%S %Z")

"""Forecast evaluation view for the Streamlit frontend."""

from typing import Any, Final, cast

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIClient,
    APIClientError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import render_centered_dataframe

FORECAST_MODELS_PATH: Final = "/api/v1/forecast/models"

FORECAST_SUMMARY_PATH: Final = "/api/v1/forecast/summary"

FORECAST_TICKER_PATH: Final = "/api/v1/forecast/tickers/{ticker}"

RANDOM_FOREST_MODEL_NAME: Final = "random_forest"


def _as_mapping_list(
    value: Any,
) -> list[dict[str, Any]]:
    """Normalize a JSON list of objects."""
    if not isinstance(
        value,
        list,
    ):
        return []

    result: list[dict[str, Any]] = []

    for item in value:
        if isinstance(
            item,
            dict,
        ):
            result.append(
                cast(
                    dict[str, Any],
                    item,
                )
            )

    return result


def _as_string_list(
    value: Any,
) -> list[str]:
    """Normalize a JSON list of strings."""
    if not isinstance(
        value,
        list,
    ):
        return []

    return [
        item
        for item in value
        if isinstance(
            item,
            str,
        )
    ]


def _as_float(
    value: Any,
    *,
    default: float = 0.0,
) -> float:
    """Convert a JSON value to float."""
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _as_int(
    value: Any,
    *,
    default: int = 0,
) -> int:
    """Convert a JSON value to int."""
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _format_decimal(
    value: Any,
    *,
    digits: int = 4,
) -> str:
    """Format a numeric value."""
    return f"{_as_float(value):.{digits}f}"


def _format_percentage(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    """Format a decimal value as a percentage."""
    return f"{_as_float(value):.{digits}%}"


def _find_model(
    models: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any] | None:
    """Find a model entry by API identifier."""
    for model in models:
        if model.get("model_name") == model_name:
            return model

    return None


def _build_metrics_frame(
    models: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Build forecast comparison table."""
    rows: list[dict[str, object]] = []

    for model in models:
        metrics = model.get("metrics")

        if not isinstance(
            metrics,
            dict,
        ):
            continue

        rows.append(
            {
                translate("forecast.column.model", language): str(
                    model.get(
                        "display_name",
                        model.get(
                            "model_name",
                            "—",
                        ),
                    )
                ),
                translate("forecast.column.observations", language): _as_int(
                    metrics.get("observations")
                ),
                "MAE": _as_float(metrics.get("mae")),
                "RMSE": _as_float(metrics.get("rmse")),
                "R²": _as_float(metrics.get("r2")),
                translate("forecast.column.direction", language): (
                    _as_float(metrics.get("directional_accuracy"))
                ),
            }
        )

    return pd.DataFrame(rows)


def _format_metrics_frame(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Create a readable copy of forecast metrics."""
    display_frame = frame.copy()

    for column in [
        "MAE",
        "RMSE",
    ]:
        if column in (display_frame.columns):
            display_frame[column] = [
                f"{_as_float(value):.5f}" for value in (display_frame[column].tolist())
            ]

    if "R²" in display_frame.columns:
        display_frame["R²"] = [
            f"{_as_float(value):.4f}" for value in (display_frame["R²"].tolist())
        ]

    direction_column = translate("forecast.column.direction", language)

    if direction_column in display_frame.columns:
        display_frame[direction_column] = [
            _format_percentage(value)
            for value in (display_frame[direction_column].tolist())
        ]

    return display_frame


def _render_model_availability(
    models_payload: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render availability of forecast artifacts."""
    models = _as_mapping_list(models_payload.get("models"))

    if not models:
        st.warning(translate("forecast.models.empty", language))
        return

    st.subheader(translate("forecast.models.heading", language))

    columns = st.columns(len(models))

    for column, model in zip(
        columns,
        models,
        strict=True,
    ):
        display_name = str(
            model.get(
                "display_name",
                model.get(
                    "model_name",
                    "—",
                ),
            )
        )

        artifact_available = bool(
            model.get(
                "artifact_available",
                False,
            )
        )

        with column:
            st.metric(
                display_name,
                translate(
                    (
                        "forecast.models.available"
                        if artifact_available
                        else "forecast.models.missing"
                    ),
                    language,
                ),
            )


def _render_random_forest_metrics(
    ticker_payload: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render main Random Forest validation metrics."""
    models = _as_mapping_list(ticker_payload.get("models"))

    random_forest = _find_model(
        models,
        RANDOM_FOREST_MODEL_NAME,
    )

    if random_forest is None:
        st.warning(translate("forecast.rf.empty", language))
        return

    metrics = random_forest.get("metrics")

    if not isinstance(
        metrics,
        dict,
    ):
        st.warning(translate("forecast.rf.metrics_empty", language))
        return

    (
        mae_column,
        rmse_column,
        r2_column,
        direction_column,
    ) = st.columns(4)

    with mae_column:
        st.metric(
            "MAE",
            _format_decimal(
                metrics.get("mae"),
                digits=5,
            ),
        )

    with rmse_column:
        st.metric(
            "RMSE",
            _format_decimal(
                metrics.get("rmse"),
                digits=5,
            ),
        )

    with r2_column:
        st.metric(
            "R²",
            _format_decimal(
                metrics.get("r2"),
            ),
        )

    with direction_column:
        st.metric(
            translate("forecast.column.direction", language),
            _format_percentage(metrics.get("directional_accuracy")),
        )


def _render_ticker_comparison(
    ticker_payload: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render model comparison for selected ticker."""
    models = _as_mapping_list(ticker_payload.get("models"))

    frame = _build_metrics_frame(models, language)

    if frame.empty:
        st.warning(translate("forecast.ticker.metrics_empty", language))
        return

    render_centered_dataframe(
        _format_metrics_frame(frame, language),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("forecast.ticker.comparison_note", language))


def _render_global_comparison(
    summary: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render aggregate validation metrics across tickers."""
    st.subheader(translate("forecast.global.heading", language))

    models = _as_mapping_list(summary.get("models"))

    frame = _build_metrics_frame(models, language)

    if frame.empty:
        st.warning(translate("forecast.global.empty", language))
        return

    render_centered_dataframe(
        _format_metrics_frame(frame, language),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("forecast.global.caption", language))


def render_forecast_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render forecast model evaluation."""
    st.title(translate("forecast.title", language))

    st.caption(translate("forecast.caption", language))

    st.info(translate("forecast.info", language))

    try:
        models_payload = client.get(FORECAST_MODELS_PATH)

        summary = client.get(FORECAST_SUMMARY_PATH)
    except APIClientError as exc:
        st.error(translate("forecast.error.load", language))

        st.caption(str(exc))

        return

    _render_model_availability(models_payload, language)

    st.divider()

    evaluation_split = str(
        summary.get(
            "evaluation_split",
            "—",
        )
    )

    target_definition = str(
        summary.get(
            "target_definition",
            "—",
        )
    )

    tickers = _as_string_list(summary.get("tickers"))

    (
        split_column,
        assets_column,
        target_column,
    ) = st.columns(
        [
            1,
            1,
            2,
        ]
    )

    with split_column:
        st.metric(
            translate("forecast.metric.split", language),
            translate(f"common.split.{evaluation_split}", language)
            if evaluation_split in {"training", "validation"}
            else evaluation_split,
        )

    with assets_column:
        st.metric(
            translate("forecast.metric.assets", language),
            str(len(tickers)),
        )

    with target_column:
        st.metric(
            translate("forecast.metric.target", language),
            target_definition,
        )

    st.divider()

    _render_global_comparison(summary, language)

    st.divider()

    st.subheader(translate("forecast.ticker.heading", language))

    if not tickers:
        st.warning(translate("forecast.ticker.empty", language))
        return

    default_index = tickers.index("SPY") if "SPY" in tickers else 0

    selected_ticker = str(
        st.selectbox(
            translate("forecast.ticker.select", language),
            tickers,
            index=default_index,
            key="forecast_ticker",
        )
    )

    try:
        ticker_payload = client.get(FORECAST_TICKER_PATH.format(ticker=selected_ticker))
    except APIClientError as exc:
        st.error(translate("forecast.ticker.error", language))

        st.caption(str(exc))

        return

    ticker_split = str(
        ticker_payload.get(
            "evaluation_split",
            "—",
        )
    )

    ticker_target = str(
        ticker_payload.get(
            "target_definition",
            "—",
        )
    )

    translated_ticker_split = (
        translate(f"common.split.{ticker_split}", language)
        if ticker_split in {"training", "validation"}
        else ticker_split
    )

    st.caption(
        translate(
            "forecast.ticker.context",
            language,
            ticker=selected_ticker,
            split=translated_ticker_split,
            target=ticker_target,
        )
    )

    _render_random_forest_metrics(ticker_payload, language)

    st.markdown(f"#### {translate('forecast.ticker.models_heading', language)}")

    _render_ticker_comparison(ticker_payload, language)

    st.divider()

    with st.expander(translate("forecast.interpretation.heading", language)):
        st.markdown(translate("forecast.interpretation.body", language))

    st.warning(translate("forecast.warning", language))

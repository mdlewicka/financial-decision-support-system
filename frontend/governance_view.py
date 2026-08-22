"""Explainability and governance view for the Streamlit frontend."""

from typing import Any, Final, cast

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.accessibility import render_chart_description
from frontend.api_client import (
    APIClient,
    APIClientError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import render_centered_dataframe

XAI_STATUS_PATH: Final = "/api/v1/xai/status"
XAI_SUMMARY_PATH: Final = "/api/v1/xai/summary"
XAI_LOCAL_PATH: Final = "/api/v1/xai/local"
XAI_FEATURE_PATH: Final = "/api/v1/xai/features/{feature_name}"

FEATURE_NAMES: Final[tuple[str, ...]] = (
    "adjusted_close_to_ma_20",
    "momentum_20",
    "adjusted_close_to_ma_5",
    "open_to_close_return",
    "daily_return",
    "volatility_5",
    "volatility_20",
    "intraday_range",
    "momentum_5",
    "volume_change",
)

GLOBAL_SHAP_CHART_HEIGHT: Final = 430
EFFECT_CHART_HEIGHT: Final = 360
LOCAL_SHAP_CHART_HEIGHT: Final = 430


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


def _format_decimal(
    value: Any,
    *,
    digits: int = 6,
) -> str:
    """Format a numeric value."""
    return f"{_as_float(value):.{digits}f}"


def _format_percentage(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    """Format a decimal value as percentage."""
    return f"{_as_float(value):.{digits}%}"


def _feature_label(
    feature_name: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return a readable feature label."""
    if feature_name.startswith("ticker_"):
        ticker = feature_name.removeprefix("ticker_")

        return translate("governance.feature.ticker", language, ticker=ticker)

    if feature_name in FEATURE_NAMES:
        return translate(f"governance.feature.{feature_name}", language)

    return feature_name.replace("_", " ")


def _localized_columns(language: LanguageCode) -> dict[str, str]:
    """Map stable internal XAI columns to presentation labels."""
    return {
        "Cecha": translate("governance.column.feature", language),
        "Identyfikator": translate("governance.column.identifier", language),
        "Średnia |SHAP|": translate("governance.column.mean_shap", language),
        "PDP/ALE": translate("governance.column.effect", language),
        "Wartość cechy": translate("governance.column.feature_value", language),
    }


def _configure_figure(
    figure: go.Figure,
    *,
    height: int,
    x_title: str,
    y_title: str,
) -> go.Figure:
    """Apply common analytical chart layout."""
    figure.update_layout(
        height=height,
        margin={
            "l": 20,
            "r": 20,
            "t": 20,
            "b": 20,
        },
        xaxis_title=x_title,
        yaxis_title=y_title,
        hovermode="closest",
    )

    return figure


def _build_global_shap_figure(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build global SHAP importance figure."""
    chart_frame = (
        frame[
            [
                "Cecha",
                "Średnia |SHAP|",
            ]
        ]
        .head(10)
        .copy()
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_frame["Średnia |SHAP|"],
                y=chart_frame["Cecha"],
                orientation="h",
                customdata=chart_frame["Średnia |SHAP|"],
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{translate('governance.column.mean_shap', language)}: "
                    "%{customdata:.6f}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=GLOBAL_SHAP_CHART_HEIGHT,
        x_title=translate("governance.chart.mean_shap", language),
        y_title=translate("governance.column.feature", language),
    )

    figure.update_yaxes(autorange="reversed")

    return figure


def _build_partial_dependence_figure(
    partial_dependence: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build Partial Dependence Plot."""
    feature_values = [
        _as_float(point.get("feature_value")) for point in partial_dependence
    ]

    average_predictions = [
        _as_float(point.get("average_prediction")) for point in partial_dependence
    ]

    figure = go.Figure(
        data=[
            go.Scatter(
                x=feature_values,
                y=average_predictions,
                mode="lines+markers",
                name="PDP",
                hovertemplate=(
                    f"{translate('governance.chart.feature_value', language)}: %{{x:.6f}}"
                    f"<br>{translate('governance.chart.average_prediction', language)}: "
                    "%{y:.6f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=EFFECT_CHART_HEIGHT,
        x_title=translate("governance.chart.feature_value", language),
        y_title=translate("governance.chart.average_prediction", language),
    )

    figure.update_layout(showlegend=False)

    return figure


def _build_ale_figure(
    accumulated_local_effects: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build Accumulated Local Effects figure."""
    bin_centers = [
        _as_float(point.get("bin_center")) for point in accumulated_local_effects
    ]

    accumulated_effects = [
        _as_float(point.get("accumulated_effect"))
        for point in accumulated_local_effects
    ]

    figure = go.Figure(
        data=[
            go.Scatter(
                x=bin_centers,
                y=accumulated_effects,
                mode="lines+markers",
                name="ALE",
                hovertemplate=(
                    f"{translate('governance.chart.bin_center', language)}: %{{x:.6f}}"
                    f"<br>{translate('governance.chart.ale_effect', language)}: "
                    "%{y:.6f}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=EFFECT_CHART_HEIGHT,
        x_title=translate("governance.chart.bin_center", language),
        y_title=translate("governance.chart.ale_effect", language),
    )

    figure.update_layout(showlegend=False)

    figure.add_hline(
        y=0,
        line_dash="dot",
    )

    return figure


def _build_local_shap_figure(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build signed local SHAP contribution figure."""
    chart_frame = (
        frame[
            [
                "Cecha",
                "SHAP",
            ]
        ]
        .head(10)
        .copy()
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_frame["SHAP"],
                y=chart_frame["Cecha"],
                orientation="h",
                customdata=chart_frame["SHAP"],
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{translate('governance.chart.shap_value', language)}: "
                    "%{customdata:.6f}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=LOCAL_SHAP_CHART_HEIGHT,
        x_title=translate("governance.chart.shap_value", language),
        y_title=translate("governance.column.feature", language),
    )

    figure.update_yaxes(autorange="reversed")

    figure.add_vline(
        x=0,
        line_dash="dot",
    )

    return figure


def _render_governance_metadata(
    status: dict[str, Any],
    summary: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render model-governance metadata."""
    model_name = str(
        summary.get(
            "model_name",
            status.get(
                "model_name",
                "—",
            ),
        )
    )

    explanation_split = str(
        summary.get(
            "explanation_split",
            "—",
        )
    )

    target_definition = str(
        summary.get(
            "target_definition",
            "—",
        )
    )

    global_observations = summary.get("global_observations")

    methods = _as_string_list(summary.get("methods"))

    (
        model_column,
        split_column,
        observations_column,
        artifact_column,
    ) = st.columns(4)

    with model_column:
        st.metric(
            translate("governance.metric.model", language),
            model_name,
        )

    with split_column:
        st.metric(
            translate("governance.metric.split", language),
            translate(f"common.split.{explanation_split}", language)
            if explanation_split in {"training", "validation"}
            else explanation_split,
        )

    with observations_column:
        st.metric(
            translate("governance.metric.observations", language),
            str(global_observations if global_observations is not None else "—"),
        )

    with artifact_column:
        st.metric(
            translate("governance.metric.artifact", language),
            (
                translate("governance.status.available", language)
                if bool(
                    status.get(
                        "artifact_available",
                        False,
                    )
                )
                else translate("governance.status.missing", language)
            ),
        )

    st.caption(translate("governance.target", language, target=target_definition))

    if methods:
        st.caption(
            translate("governance.methods", language, methods=", ".join(methods))
        )


def _render_global_explainability(
    summary: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render global SHAP importance."""
    st.subheader(translate("governance.global.heading", language))

    features = _as_mapping_list(summary.get("features"))

    if not features:
        st.warning(translate("governance.global.empty", language))
        return

    rows: list[dict[str, object]] = []

    for feature in features:
        feature_name = str(
            feature.get(
                "feature",
                "",
            )
        )

        rows.append(
            {
                "Cecha": _feature_label(feature_name, language),
                "Identyfikator": (feature_name),
                "Średnia |SHAP|": (_as_float(feature.get("mean_absolute_shap_value"))),
                "PDP/ALE": (
                    translate("governance.yes", language)
                    if bool(
                        feature.get(
                            "effect_available",
                            False,
                        )
                    )
                    else translate("governance.no", language)
                ),
            }
        )

    frame = pd.DataFrame(rows)

    display_frame = frame.drop(
        columns=[
            "Identyfikator",
        ]
    )

    render_centered_dataframe(
        display_frame.rename(columns=_localized_columns(language)),
        width="stretch",
        hide_index=True,
    )

    figure = _build_global_shap_figure(frame, language)

    st.plotly_chart(
        figure,
        width="stretch",
    )
    render_chart_description(
        translate("accessibility.governance.global", language),
        language,
    )

    st.warning(translate("governance.global.warning", language))


def _render_feature_effect(
    client: APIClient,
    effect_features: list[str],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render PDP and ALE for one feature."""
    st.subheader(translate("governance.effect.heading", language))

    if not effect_features:
        st.info(translate("governance.effect.empty", language))
        return

    selected_feature = str(
        st.selectbox(
            translate("governance.effect.select", language),
            effect_features,
            format_func=lambda value: _feature_label(value, language),
            key=("governance_effect_feature"),
        )
    )

    try:
        payload = client.get(XAI_FEATURE_PATH.format(feature_name=(selected_feature)))
    except APIClientError as exc:
        st.error(translate("governance.effect.error", language))

        st.caption(str(exc))
        return

    st.metric(
        translate("governance.column.mean_shap", language),
        _format_decimal(payload.get("mean_absolute_shap_value")),
    )

    partial_dependence = _as_mapping_list(payload.get("partial_dependence"))

    accumulated_local_effects = _as_mapping_list(
        payload.get("accumulated_local_effects")
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("#### Partial Dependence")

        if partial_dependence:
            figure = _build_partial_dependence_figure(partial_dependence, language)

            st.plotly_chart(
                figure,
                width="stretch",
            )
            render_chart_description(
                translate("accessibility.governance.pdp", language),
                language,
            )
        else:
            st.info(translate("governance.effect.pdp_empty", language))

    with right_column:
        st.markdown("#### Accumulated Local Effects")

        if accumulated_local_effects:
            figure = _build_ale_figure(accumulated_local_effects, language)

            st.plotly_chart(
                figure,
                width="stretch",
            )
            render_chart_description(
                translate("accessibility.governance.ale", language),
                language,
            )
        else:
            st.info(translate("governance.effect.ale_empty", language))

    with st.expander(translate("governance.effect.source", language)):
        source_left, source_right = st.columns(2)

        with source_left:
            st.write("Partial Dependence")

            render_centered_dataframe(
                pd.DataFrame(partial_dependence),
                width="stretch",
                hide_index=True,
            )

        with source_right:
            st.write("Accumulated Local Effects")

            render_centered_dataframe(
                pd.DataFrame(accumulated_local_effects),
                width="stretch",
                hide_index=True,
            )


def _render_local_explainability(
    local_payload: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render representative local SHAP explanation."""
    st.subheader(translate("governance.local.heading", language))

    st.caption(translate("governance.local.caption", language))

    ticker = str(
        local_payload.get(
            "ticker",
            "—",
        )
    )

    feature_date = str(
        local_payload.get(
            "feature_date",
            "—",
        )
    )

    target_date = str(
        local_payload.get(
            "target_date",
            "—",
        )
    )

    st.write(
        translate(
            "governance.local.context",
            language,
            ticker=ticker,
            feature_date=feature_date,
            target_date=target_date,
        )
    )

    (
        actual_column,
        predicted_column,
        base_column,
        reconstructed_column,
    ) = st.columns(4)

    with actual_column:
        st.metric(
            translate("governance.local.actual", language),
            _format_percentage(local_payload.get("actual_return")),
        )

    with predicted_column:
        st.metric(
            translate("governance.local.predicted", language),
            _format_percentage(local_payload.get("predicted_return")),
        )

    with base_column:
        st.metric(
            translate("governance.local.base", language),
            _format_percentage(local_payload.get("base_value")),
        )

    with reconstructed_column:
        st.metric(
            translate("governance.local.reconstructed", language),
            _format_percentage(local_payload.get("reconstructed_prediction")),
        )

    features = _as_mapping_list(local_payload.get("features"))

    if not features:
        st.warning(translate("governance.local.empty", language))
        return

    sorted_features = sorted(
        features,
        key=lambda item: abs(_as_float(item.get("shap_value"))),
        reverse=True,
    )

    rows: list[dict[str, object]] = []

    for feature in sorted_features:
        feature_name = str(
            feature.get(
                "feature",
                "",
            )
        )

        rows.append(
            {
                "Cecha": _feature_label(feature_name, language),
                "Wartość cechy": (_as_float(feature.get("feature_value"))),
                "SHAP": (_as_float(feature.get("shap_value"))),
            }
        )

    frame = pd.DataFrame(rows)

    render_centered_dataframe(
        frame.rename(columns=_localized_columns(language)),
        width="stretch",
        hide_index=True,
    )

    figure = _build_local_shap_figure(frame, language)

    st.plotly_chart(
        figure,
        width="stretch",
    )
    render_chart_description(
        translate("accessibility.governance.local", language),
        language,
    )


def _render_governance_section(
    summary: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render evaluation governance information."""
    st.subheader(translate("governance.section.heading", language))

    st.markdown(translate("governance.section.body", language))

    explanation_split = str(
        summary.get(
            "explanation_split",
            "—",
        )
    )

    if explanation_split == "validation":
        st.success(translate("governance.section.valid", language))
    else:
        st.warning(translate("governance.section.invalid", language))


def render_governance_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render explainability and model governance."""
    st.title(translate("governance.title", language))

    st.caption(translate("governance.caption", language))

    try:
        status = client.get(XAI_STATUS_PATH)

        summary = client.get(XAI_SUMMARY_PATH)
    except APIClientError as exc:
        st.error(translate("governance.error.load", language))

        st.caption(str(exc))

        return

    if not bool(
        status.get(
            "artifact_available",
            False,
        )
    ):
        st.warning(translate("governance.artifact.missing", language))
        return

    _render_governance_metadata(
        status,
        summary,
        language,
    )

    st.divider()

    _render_global_explainability(summary, language)

    st.divider()

    effect_features = _as_string_list(summary.get("effect_features"))

    _render_feature_effect(
        client,
        effect_features,
        language,
    )

    st.divider()

    try:
        local_payload = client.get(XAI_LOCAL_PATH)
    except APIClientError as exc:
        st.error(translate("governance.local.error", language))

        st.caption(str(exc))
    else:
        _render_local_explainability(local_payload, language)

    st.divider()

    _render_governance_section(summary, language)

    with st.expander(translate("governance.limitations.heading", language)):
        st.markdown(translate("governance.limitations.body", language))

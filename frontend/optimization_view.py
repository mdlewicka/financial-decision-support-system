"""Portfolio optimization view for the Streamlit frontend."""

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

PORTFOLIO_SCENARIOS_PATH: Final = "/api/v1/portfolio/scenarios"

PORTFOLIO_SCENARIO_PATH: Final = "/api/v1/portfolio/scenarios/{scenario_name}"

PORTFOLIO_RETURNS_PATH: Final = "/api/v1/portfolio/scenarios/{scenario_name}/returns"

DEFAULT_SCENARIO: Final = "maximum_sharpe"

WEIGHTS_CHART_HEIGHT: Final = 430
CONTRIBUTION_CHART_HEIGHT: Final = 390
CUMULATIVE_RETURN_CHART_HEIGHT: Final = 430


def _as_mapping(
    value: Any,
) -> dict[str, Any]:
    """Normalize a JSON object."""
    if not isinstance(
        value,
        dict,
    ):
        return {}

    return cast(
        dict[str, Any],
        value,
    )


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
    """Convert JSON value to float."""
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
    """Convert JSON value to int."""
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _format_percentage(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    """Format decimal value as percentage."""
    return f"{_as_float(value):.{digits}%}"


def _format_decimal(
    value: Any,
    *,
    digits: int = 3,
) -> str:
    """Format numeric value."""
    return f"{_as_float(value):.{digits}f}"


def _scenario_label(
    scenario_name: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return readable scenario name."""
    try:
        return translate(f"decision.scenario.{scenario_name}", language)
    except KeyError:
        return scenario_name.replace("_", " ").title()


def _localized_columns(language: LanguageCode) -> dict[str, str]:
    """Map stable internal table columns to presentation labels."""
    return {
        "Strategia": translate("optimization.column.strategy", language),
        "Zwrot roczny": translate("risk.column.return", language),
        "Zmienność roczna": translate("risk.column.volatility", language),
        "Sharpe": translate("risk.column.sharpe", language),
        "Sortino": translate("risk.column.sortino", language),
        "Max drawdown": translate("risk.column.drawdown", language),
        "Koncentracja": translate("risk.column.concentration", language),
        "Efektywna liczba aktywów": translate("risk.column.effective_assets", language),
        "Diversification ratio": translate("risk.column.diversification", language),
        "Liczba aktywów": translate("optimization.column.asset_count", language),
        "Instrument": translate("optimization.column.instrument", language),
        "Waga": translate("optimization.column.weight", language),
        "Wkład w zwrot roczny": translate(
            "optimization.column.return_contribution", language
        ),
        "Wkład w zmienność roczną": translate(
            "optimization.column.volatility_contribution", language
        ),
        "Zbiór": translate("risk.column.split", language),
        "Obserwacje": translate("risk.column.observations", language),
        "Data": translate("optimization.column.date", language),
        "Zwrot portfela": translate("optimization.column.portfolio_return", language),
        "Zwrot skumulowany": translate(
            "optimization.column.cumulative_return", language
        ),
    }


def _find_scenario(
    scenarios: list[dict[str, Any]],
    scenario_name: str,
) -> dict[str, Any] | None:
    """Find scenario by API identifier."""
    for scenario in scenarios:
        if scenario.get("name") == scenario_name:
            return scenario

    return None


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
        showlegend=False,
    )

    return figure


def _build_weights_figure(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build portfolio-weight figure."""
    chart_frame = frame.sort_values(
        by="Waga",
        ascending=False,
    ).copy()

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_frame["Waga"],
                y=chart_frame["Instrument"],
                orientation="h",
                customdata=chart_frame["Waga"],
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{translate('optimization.column.weight', language)}: "
                    "%{customdata:.2%}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=WEIGHTS_CHART_HEIGHT,
        x_title=translate("optimization.chart.weight", language),
        y_title=translate("optimization.column.instrument", language),
    )

    figure.update_xaxes(tickformat=".0%")

    figure.update_yaxes(autorange="reversed")

    return figure


def _build_contribution_figure(
    frame: pd.DataFrame,
    *,
    contribution_column: str,
    x_title: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build asset contribution figure."""
    chart_frame = frame.copy()

    chart_frame["_absolute_contribution"] = chart_frame[contribution_column].abs()

    chart_frame = chart_frame.sort_values(
        by="_absolute_contribution",
        ascending=False,
    )

    figure = go.Figure(
        data=[
            go.Bar(
                x=chart_frame[contribution_column],
                y=chart_frame["Instrument"],
                orientation="h",
                customdata=chart_frame[contribution_column],
                hovertemplate=(
                    f"<b>%{{y}}</b><br>{translate('optimization.chart.contribution', language)}: "
                    "%{customdata:.2%}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=CONTRIBUTION_CHART_HEIGHT,
        x_title=x_title,
        y_title=translate("optimization.column.instrument", language),
    )

    figure.update_xaxes(tickformat=".1%")

    figure.update_yaxes(autorange="reversed")

    figure.add_vline(
        x=0,
        line_dash="dot",
    )

    return figure


def _build_cumulative_return_figure(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> go.Figure:
    """Build cumulative portfolio return figure."""
    figure = go.Figure(
        data=[
            go.Scatter(
                x=frame["Data"],
                y=frame["Zwrot skumulowany"],
                mode="lines",
                name=translate("optimization.column.cumulative_return", language),
                hovertemplate=(
                    f"{translate('optimization.column.date', language)}: %{{x|%Y-%m-%d}}"
                    f"<br>{translate('optimization.column.cumulative_return', language)}: "
                    "%{y:.2%}<extra></extra>"
                ),
            )
        ]
    )

    _configure_figure(
        figure,
        height=(CUMULATIVE_RETURN_CHART_HEIGHT),
        x_title=translate("optimization.column.date", language),
        y_title=translate("optimization.column.cumulative_return", language),
    )

    figure.update_yaxes(tickformat=".0%")

    figure.add_hline(
        y=0,
        line_dash="dot",
    )

    return figure


def _build_scenario_comparison(
    scenarios: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Build portfolio strategy comparison table."""
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        validation = _as_mapping(scenario.get("validation"))

        scenario_name = str(
            scenario.get(
                "name",
                "",
            )
        )

        weights = _as_mapping_list(scenario.get("weights"))

        rows.append(
            {
                "Strategia": _scenario_label(scenario_name, language),
                "Zwrot roczny": (_as_float(validation.get("annualized_mean_return"))),
                "Zmienność roczna": (
                    _as_float(validation.get("annualized_volatility"))
                ),
                "Sharpe": (_as_float(validation.get("sharpe_ratio"))),
                "Sortino": (_as_float(validation.get("sortino_ratio"))),
                "Max drawdown": (_as_float(validation.get("maximum_drawdown"))),
                "Koncentracja": (_as_float(validation.get("concentration_index"))),
                "Efektywna liczba aktywów": (
                    _as_float(validation.get("effective_number_of_assets"))
                ),
                "Diversification ratio": (
                    _as_float(validation.get("diversification_ratio"))
                ),
                "Liczba aktywów": len(weights),
            }
        )

    return pd.DataFrame(rows)


def _format_scenario_comparison(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Format portfolio comparison table."""
    display_frame = frame.copy()

    percentage_columns = [
        "Zwrot roczny",
        "Zmienność roczna",
        "Max drawdown",
    ]

    for column in percentage_columns:
        if column in (display_frame.columns):
            display_frame[column] = [
                _format_percentage(value) for value in (display_frame[column].tolist())
            ]

    decimal_columns = [
        "Sharpe",
        "Sortino",
        "Koncentracja",
        "Efektywna liczba aktywów",
        "Diversification ratio",
    ]

    for column in decimal_columns:
        if column in (display_frame.columns):
            display_frame[column] = [
                _format_decimal(value) for value in (display_frame[column].tolist())
            ]

    return display_frame


def _build_weights_frame(
    detail: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    """Build asset-weight table."""
    weights = _as_mapping_list(detail.get("weights"))

    rows: list[dict[str, object]] = []

    for item in weights:
        rows.append(
            {
                "Instrument": str(
                    item.get(
                        "ticker",
                        "—",
                    )
                ),
                "Waga": (_as_float(item.get("weight"))),
            }
        )

    frame = pd.DataFrame(rows)

    if not frame.empty:
        frame = frame.sort_values(
            by="Waga",
            ascending=False,
        ).reset_index(drop=True)

    return frame


def _build_contributions_frame(
    detail: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    """Build validation contribution table."""
    contributions = _as_mapping_list(detail.get("validation_contributions"))

    rows: list[dict[str, object]] = []

    for item in contributions:
        rows.append(
            {
                "Instrument": str(
                    item.get(
                        "ticker",
                        "—",
                    )
                ),
                "Waga": (_as_float(item.get("weight"))),
                "Wkład w zwrot roczny": (
                    _as_float(item.get("annualized_return_contribution"))
                ),
                "Wkład w zmienność roczną": (
                    _as_float(item.get("annualized_volatility_contribution"))
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_split_comparison(
    detail: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    """Compare training and validation portfolio metrics."""
    rows: list[dict[str, object]] = []

    for split_name in [
        "training",
        "validation",
    ]:
        block = _as_mapping(detail.get(split_name))

        rows.append(
            {
                "Zbiór": (split_name),
                "Obserwacje": (_as_int(block.get("observations"))),
                "Zwrot roczny": (
                    _format_percentage(block.get("annualized_mean_return"))
                ),
                "Zmienność roczna": (
                    _format_percentage(block.get("annualized_volatility"))
                ),
                "Sharpe": (_format_decimal(block.get("sharpe_ratio"))),
                "Sortino": (_format_decimal(block.get("sortino_ratio"))),
                "Max drawdown": (_format_percentage(block.get("maximum_drawdown"))),
                "Koncentracja": (_format_decimal(block.get("concentration_index"))),
                "Efektywna liczba aktywów": (
                    _format_decimal(block.get("effective_number_of_assets"))
                ),
                "Diversification ratio": (
                    _format_decimal(block.get("diversification_ratio"))
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_returns_frame(
    returns_payload: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    """Build portfolio return time series."""
    returns = _as_mapping_list(returns_payload.get("returns"))

    rows: list[dict[str, object]] = []

    for item in returns:
        rows.append(
            {
                "Data": str(
                    item.get(
                        "date",
                        "",
                    )
                ),
                "Zwrot portfela": (_as_float(item.get("portfolio_return"))),
                "Zwrot skumulowany": (_as_float(item.get("cumulative_return"))),
            }
        )

    frame = pd.DataFrame(rows)

    if frame.empty:
        return frame

    frame["Data"] = pd.to_datetime(
        frame["Data"],
        errors="coerce",
    )

    return frame.dropna(
        subset=[
            "Data",
        ]
    ).sort_values(by="Data")


def _render_validation_metrics(
    detail: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render key portfolio validation metrics."""
    validation = _as_mapping(detail.get("validation"))

    (
        return_column,
        volatility_column,
        sharpe_column,
        drawdown_column,
    ) = st.columns(4)

    with return_column:
        st.metric(
            translate("risk.metric.return", language),
            _format_percentage(validation.get("annualized_mean_return")),
        )

    with volatility_column:
        st.metric(
            translate("risk.metric.volatility", language),
            _format_percentage(validation.get("annualized_volatility")),
        )

    with sharpe_column:
        st.metric(
            translate("risk.metric.sharpe", language),
            _format_decimal(validation.get("sharpe_ratio")),
        )

    with drawdown_column:
        st.metric(
            translate("risk.metric.drawdown", language),
            _format_percentage(validation.get("maximum_drawdown")),
        )

    (
        sortino_column,
        concentration_column,
        effective_column,
        diversification_column,
    ) = st.columns(4)

    with sortino_column:
        st.metric(
            translate("risk.column.sortino", language),
            _format_decimal(validation.get("sortino_ratio")),
        )

    with concentration_column:
        st.metric(
            translate("risk.column.concentration", language),
            _format_decimal(validation.get("concentration_index")),
        )

    with effective_column:
        st.metric(
            translate("risk.metric.effective_assets", language),
            _format_decimal(validation.get("effective_number_of_assets")),
        )

    with diversification_column:
        st.metric(
            translate("risk.metric.diversification", language),
            _format_decimal(validation.get("diversification_ratio")),
        )


def _render_weights(
    detail: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render portfolio weights."""
    st.subheader(translate("optimization.weights.heading", language))

    frame = _build_weights_frame(detail)

    if frame.empty:
        st.warning(translate("optimization.weights.empty", language))
        return

    display_frame = frame.copy()

    display_frame["Waga"] = [
        _format_percentage(value) for value in (display_frame["Waga"].tolist())
    ]

    left_column, right_column = st.columns(
        [
            1,
            2,
        ]
    )

    with left_column:
        render_centered_dataframe(
            display_frame.rename(columns=_localized_columns(language)),
            width="stretch",
            hide_index=True,
        )

    with right_column:
        figure = _build_weights_figure(frame, language)

        st.plotly_chart(
            figure,
            width="stretch",
        )
        render_chart_description(
            translate("accessibility.optimization.weights", language),
            language,
        )


def _render_contributions(
    detail: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render asset return and volatility contributions."""
    st.subheader(translate("optimization.contributions.heading", language))

    frame = _build_contributions_frame(detail)

    if frame.empty:
        st.info(translate("optimization.contributions.empty", language))
        return

    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown(f"#### {translate('optimization.contributions.return', language)}")

        return_figure = _build_contribution_figure(
            frame,
            contribution_column=("Wkład w zwrot roczny"),
            x_title=translate("optimization.column.return_contribution", language),
            language=language,
        )

        st.plotly_chart(
            return_figure,
            width="stretch",
        )
        render_chart_description(
            translate("accessibility.optimization.return_contribution", language),
            language,
        )

    with right_column:
        st.markdown(
            f"#### {translate('optimization.contributions.volatility', language)}"
        )

        volatility_figure = _build_contribution_figure(
            frame,
            contribution_column=("Wkład w zmienność roczną"),
            x_title=translate("optimization.column.volatility_contribution", language),
            language=language,
        )

        st.plotly_chart(
            volatility_figure,
            width="stretch",
        )
        render_chart_description(
            translate("accessibility.optimization.volatility_contribution", language),
            language,
        )

    display_frame = frame.copy()

    for column in [
        "Waga",
        "Wkład w zwrot roczny",
        "Wkład w zmienność roczną",
    ]:
        display_frame[column] = [
            _format_percentage(value) for value in (display_frame[column].tolist())
        ]

    with st.expander(translate("optimization.contributions.data", language)):
        render_centered_dataframe(
            display_frame.rename(columns=_localized_columns(language)),
            width="stretch",
            hide_index=True,
        )


def _render_cumulative_returns(
    returns_payload: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render validation cumulative return."""
    st.subheader(translate("optimization.cumulative.heading", language))

    frame = _build_returns_frame(returns_payload)

    if frame.empty:
        st.warning(translate("optimization.cumulative.empty", language))
        return

    figure = _build_cumulative_return_figure(frame, language)

    st.plotly_chart(
        figure,
        width="stretch",
    )
    render_chart_description(
        translate("accessibility.optimization.cumulative", language),
        language,
    )

    final_cumulative_return = _as_float(frame["Zwrot skumulowany"].iloc[-1])

    st.metric(
        translate("optimization.cumulative.final", language),
        _format_percentage(final_cumulative_return),
    )

    st.caption(translate("optimization.cumulative.caption", language))


def render_optimization_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render portfolio optimization analysis."""
    st.title(translate("optimization.title", language))

    st.caption(translate("optimization.caption", language))

    st.info(translate("optimization.info", language))

    try:
        scenarios_payload = client.get(PORTFOLIO_SCENARIOS_PATH)
    except APIClientError as exc:
        st.error(translate("optimization.error.load", language))

        st.caption(str(exc))

        return

    scenarios = _as_mapping_list(scenarios_payload.get("scenarios"))

    if not scenarios:
        st.warning(translate("optimization.empty", language))
        return

    evaluation_split = str(
        scenarios_payload.get(
            "evaluation_split",
            "—",
        )
    )

    return_definition = str(
        scenarios_payload.get(
            "return_definition",
            "—",
        )
    )

    tickers = _as_string_list(scenarios_payload.get("tickers"))

    (
        split_column,
        strategies_column,
        assets_column,
    ) = st.columns(3)

    with split_column:
        st.metric(
            translate("optimization.metric.split", language),
            translate(f"common.split.{evaluation_split}", language)
            if evaluation_split in {"training", "validation"}
            else evaluation_split,
        )

    with strategies_column:
        st.metric(
            translate("optimization.metric.strategies", language),
            str(len(scenarios)),
        )

    with assets_column:
        st.metric(
            translate("optimization.metric.assets", language),
            str(len(tickers)),
        )

    st.caption(
        translate(
            "optimization.return_definition",
            language,
            definition=return_definition,
        )
    )

    st.divider()

    st.subheader(translate("optimization.comparison.heading", language))

    comparison = _build_scenario_comparison(scenarios, language)

    render_centered_dataframe(
        _format_scenario_comparison(comparison).rename(
            columns=_localized_columns(language)
        ),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("optimization.comparison.caption", language))

    st.divider()

    scenario_names = [
        str(scenario.get("name")) for scenario in scenarios if scenario.get("name")
    ]

    default_index = (
        scenario_names.index(DEFAULT_SCENARIO)
        if DEFAULT_SCENARIO in scenario_names
        else 0
    )

    selected_scenario = str(
        st.selectbox(
            translate("optimization.scenario.select", language),
            scenario_names,
            index=default_index,
            format_func=lambda value: _scenario_label(value, language),
            key=("optimization_scenario"),
        )
    )

    selected_summary = _find_scenario(
        scenarios,
        selected_scenario,
    )

    if selected_summary is not None:
        construction = str(
            selected_summary.get(
                "construction",
                "—",
            )
        )

        st.caption(
            translate(
                "optimization.scenario.construction",
                language,
                construction=construction,
            )
        )

    try:
        detail = client.get(
            PORTFOLIO_SCENARIO_PATH.format(scenario_name=(selected_scenario))
        )

        returns_payload = client.get(
            PORTFOLIO_RETURNS_PATH.format(scenario_name=(selected_scenario))
        )
    except APIClientError as exc:
        st.error(translate("optimization.scenario.error", language))

        st.caption(str(exc))

        return

    st.subheader(translate("optimization.validation.heading", language))

    _render_validation_metrics(detail, language)

    st.divider()

    _render_weights(detail, language)

    st.divider()

    _render_contributions(detail, language)

    st.divider()

    _render_cumulative_returns(returns_payload, language)

    st.divider()

    st.subheader(translate("optimization.split.heading", language))

    render_centered_dataframe(
        _build_split_comparison(detail)
        .replace(
            {
                "training": translate("common.split.training", language),
                "validation": translate("common.split.validation", language),
            }
        )
        .rename(columns=_localized_columns(language)),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("optimization.split.caption", language))

    st.divider()

    with st.expander(translate("optimization.interpretation.heading", language)):
        st.markdown(translate("optimization.interpretation.body", language))

    with st.expander(translate("optimization.limitations.heading", language)):
        st.markdown(translate("optimization.limitations.body", language))

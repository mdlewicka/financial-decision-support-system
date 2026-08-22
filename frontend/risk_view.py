"""Risk and scenario analysis view for the Streamlit frontend."""

from typing import Any, Final, cast

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIClient,
    APIClientError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import render_centered_dataframe

RISK_MARKET_PATH: Final = "/api/v1/risk/market"

RISK_SCENARIOS_PATH: Final = "/api/v1/risk/scenarios"

RISK_SCENARIO_PATH: Final = "/api/v1/risk/scenarios/{scenario_name}"

DEFAULT_SCENARIO: Final = "maximum_sharpe"


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
    """Return a readable scenario name."""
    try:
        return translate(f"decision.scenario.{scenario_name}", language)
    except KeyError:
        return scenario_name.replace("_", " ").title()


def _find_scenario(
    scenarios: list[dict[str, Any]],
    scenario_name: str,
) -> dict[str, Any] | None:
    """Find scenario by API identifier."""
    for scenario in scenarios:
        if scenario.get("name") == scenario_name:
            return scenario

    return None


def _build_scenario_comparison(
    scenarios: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Build validation risk comparison table."""
    rows: list[dict[str, object]] = []

    for scenario in scenarios:
        risk = _as_mapping(scenario.get("risk"))

        metrics = _as_mapping(risk.get("metrics"))

        scenario_name = str(
            scenario.get(
                "name",
                "",
            )
        )

        rows.append(
            {
                translate("risk.column.scenario", language): _scenario_label(
                    scenario_name, language
                ),
                translate("risk.column.return", language): _as_float(
                    metrics.get("annualized_mean_return")
                ),
                translate("risk.column.volatility", language): _as_float(
                    metrics.get("annualized_volatility")
                ),
                translate("risk.column.sharpe", language): _as_float(
                    metrics.get("sharpe_ratio")
                ),
                translate("risk.column.sortino", language): _as_float(
                    metrics.get("sortino_ratio")
                ),
                translate("risk.column.var", language): _as_float(
                    metrics.get("value_at_risk")
                ),
                translate("risk.column.cvar", language): _as_float(
                    metrics.get("conditional_value_at_risk")
                ),
                translate("risk.column.drawdown", language): _as_float(
                    metrics.get("maximum_drawdown")
                ),
                translate("risk.column.concentration", language): _as_float(
                    risk.get("concentration_index")
                ),
                translate("risk.column.effective_assets", language): (
                    _as_float(risk.get("effective_number_of_assets"))
                ),
                translate("risk.column.diversification", language): _as_float(
                    risk.get("diversification_ratio")
                ),
            }
        )

    return pd.DataFrame(rows)


def _format_scenario_comparison(
    frame: pd.DataFrame,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Format scenario comparison for display."""
    display_frame = frame.copy()

    percentage_columns = [
        translate("risk.column.return", language),
        translate("risk.column.volatility", language),
        translate("risk.column.var", language),
        translate("risk.column.cvar", language),
        translate("risk.column.drawdown", language),
    ]

    for column in percentage_columns:
        if column in (display_frame.columns):
            display_frame[column] = [
                _format_percentage(value) for value in (display_frame[column].tolist())
            ]

    decimal_columns = [
        translate("risk.column.sharpe", language),
        translate("risk.column.sortino", language),
        translate("risk.column.concentration", language),
        translate("risk.column.effective_assets", language),
        translate("risk.column.diversification", language),
    ]

    for column in decimal_columns:
        if column in (display_frame.columns):
            display_frame[column] = [
                _format_decimal(value) for value in (display_frame[column].tolist())
            ]

    return display_frame


def _build_split_comparison(
    detail: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Compare training and validation risk metrics."""
    rows: list[dict[str, object]] = []

    for split_name in [
        "training",
        "validation",
    ]:
        block = _as_mapping(detail.get(split_name))

        metrics = _as_mapping(block.get("metrics"))

        rows.append(
            {
                translate("risk.column.split", language): translate(
                    f"common.split.{split_name}", language
                ),
                translate("risk.column.observations", language): _as_int(
                    metrics.get("observations")
                ),
                translate("risk.column.return", language): (
                    _format_percentage(metrics.get("annualized_mean_return"))
                ),
                translate("risk.column.volatility", language): (
                    _format_percentage(metrics.get("annualized_volatility"))
                ),
                translate("risk.column.sharpe", language): _format_decimal(
                    metrics.get("sharpe_ratio")
                ),
                translate("risk.column.sortino", language): _format_decimal(
                    metrics.get("sortino_ratio")
                ),
                translate("risk.column.var", language): _format_percentage(
                    metrics.get("value_at_risk")
                ),
                translate("risk.column.cvar", language): _format_percentage(
                    metrics.get("conditional_value_at_risk")
                ),
                translate("risk.column.drawdown", language): _format_percentage(
                    metrics.get("maximum_drawdown")
                ),
                translate("risk.column.concentration", language): _format_decimal(
                    block.get("concentration_index")
                ),
                translate("risk.column.effective_assets", language): (
                    _format_decimal(block.get("effective_number_of_assets"))
                ),
                translate("risk.column.diversification", language): (
                    _format_decimal(block.get("diversification_ratio"))
                ),
            }
        )

    return pd.DataFrame(rows)


def _build_correlation_frame(
    market: dict[
        str,
        Any,
    ],
) -> pd.DataFrame:
    """Build asset correlation matrix."""
    matrix = _as_mapping_list(market.get("correlation_matrix"))

    if not matrix:
        return pd.DataFrame()

    rows: dict[
        str,
        dict[str, float],
    ] = {}

    for item in matrix:
        ticker = str(
            item.get(
                "ticker",
                "",
            )
        )

        correlations = _as_mapping(item.get("correlations"))

        rows[ticker] = {
            str(key): _as_float(value) for key, value in (correlations.items())
        }

    return pd.DataFrame.from_dict(
        rows,
        orient="index",
    )


def _render_market_context(
    market: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render validation market context."""
    st.subheader(translate("risk.market.heading", language))

    tickers = _as_string_list(market.get("tickers"))

    (
        observations_column,
        assets_column,
        periods_column,
        dates_column,
    ) = st.columns(4)

    with observations_column:
        st.metric(
            translate("risk.market.observations", language),
            str(_as_int(market.get("observations"))),
        )

    with assets_column:
        st.metric(
            translate("risk.market.assets", language),
            str(len(tickers)),
        )

    with periods_column:
        st.metric(
            translate("risk.market.periods", language),
            str(_as_int(market.get("periods_per_year"))),
        )

    with dates_column:
        date_min = str(
            market.get(
                "date_min",
                "—",
            )
        )

        date_max = str(
            market.get(
                "date_max",
                "—",
            )
        )

        st.metric(
            translate("risk.market.range", language),
            f"{date_min} – {date_max}",
        )

    if tickers:
        st.caption(
            translate("risk.market.instruments", language, tickers=", ".join(tickers))
        )

    st.caption(
        translate(
            "risk.market.return_definition",
            language,
            definition=market.get("return_definition", "—"),
        )
    )

    correlation_frame = _build_correlation_frame(market)

    if not correlation_frame.empty:
        with st.expander(translate("risk.market.correlation", language)):
            render_centered_dataframe(
                correlation_frame.round(3),
                width="stretch",
            )


def _render_validation_metrics(
    detail: dict[
        str,
        Any,
    ],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render key validation risk metrics."""
    validation = _as_mapping(detail.get("validation"))

    metrics = _as_mapping(validation.get("metrics"))

    confidence_level = _as_float(
        metrics.get(
            "confidence_level",
            0.95,
        )
    )

    (
        return_column,
        volatility_column,
        sharpe_column,
        drawdown_column,
    ) = st.columns(4)

    with return_column:
        st.metric(
            translate("risk.metric.return", language),
            _format_percentage(metrics.get("annualized_mean_return")),
        )

    with volatility_column:
        st.metric(
            translate("risk.metric.volatility", language),
            _format_percentage(metrics.get("annualized_volatility")),
        )

    with sharpe_column:
        st.metric(
            translate("risk.metric.sharpe", language),
            _format_decimal(metrics.get("sharpe_ratio")),
        )

    with drawdown_column:
        st.metric(
            translate("risk.metric.drawdown", language),
            _format_percentage(metrics.get("maximum_drawdown")),
        )

    (
        var_column,
        cvar_column,
        effective_column,
        diversification_column,
    ) = st.columns(4)

    confidence_label = f"{confidence_level:.0%}"

    with var_column:
        st.metric(
            f"VaR ({confidence_label})",
            _format_percentage(metrics.get("value_at_risk")),
        )

    with cvar_column:
        st.metric(
            f"CVaR ({confidence_label})",
            _format_percentage(metrics.get("conditional_value_at_risk")),
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


def render_risk_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render risk and scenario analysis."""
    st.title(translate("risk.title", language))

    st.caption(translate("risk.caption", language))

    st.info(translate("risk.info", language))

    try:
        market = client.get(RISK_MARKET_PATH)

        scenarios_payload = client.get(RISK_SCENARIOS_PATH)
    except APIClientError as exc:
        st.error(translate("risk.error.load", language))

        st.caption(str(exc))

        return

    scenarios = _as_mapping_list(scenarios_payload.get("scenarios"))

    if not scenarios:
        st.warning(translate("risk.scenarios.empty", language))
        return

    st.subheader(translate("risk.comparison.heading", language))

    comparison = _build_scenario_comparison(scenarios, language)

    render_centered_dataframe(
        _format_scenario_comparison(comparison, language),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("risk.comparison.caption", language))

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
            translate("risk.scenario.select", language),
            scenario_names,
            index=default_index,
            format_func=lambda value: _scenario_label(value, language),
            key="risk_scenario",
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
                "risk.scenario.construction",
                language,
                construction=construction,
            )
        )

    try:
        detail = client.get(
            RISK_SCENARIO_PATH.format(scenario_name=(selected_scenario))
        )
    except APIClientError as exc:
        st.error(translate("risk.scenario.error", language))

        st.caption(str(exc))

        return

    st.subheader(translate("risk.profile.heading", language))

    _render_validation_metrics(detail, language)

    st.divider()

    st.subheader(translate("risk.split.heading", language))

    render_centered_dataframe(
        _build_split_comparison(detail, language),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("risk.split.caption", language))

    st.divider()

    _render_market_context(market, language)

    st.divider()

    with st.expander(translate("risk.interpretation.heading", language)):
        st.markdown(translate("risk.interpretation.body", language))

    with st.expander(translate("risk.limitations.heading", language)):
        st.markdown(translate("risk.limitations.body", language))

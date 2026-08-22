"""Analytical dashboard for the Streamlit frontend."""

from typing import Any, Final, cast

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIClient,
    APIClientError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import render_centered_dataframe

FORECAST_SUMMARY_PATH: Final = "/api/v1/forecast/summary"

RISK_SCENARIOS_PATH: Final = "/api/v1/risk/scenarios"

PORTFOLIO_SCENARIOS_PATH: Final = "/api/v1/portfolio/scenarios"

RAG_STATUS_PATH: Final = "/api/v1/rag/status"

AGENT_STATUS_PATH: Final = "/api/v1/agents/status"

BALANCED_DECISION_PATH: Final = (
    "/api/v1/decision/recommendations/balanced?available_liquidity_fraction=0.10"
)

SELECTED_PAGE_STATE_KEY: Final = "selected_application_page"

DECISION_WORKSPACE_PAGE: Final = "Decision Workspace"
FORECASTING_PAGE: Final = "Forecasting"
MARKET_INTELLIGENCE_PAGE: Final = "Market Intelligence"
REPORTS_PAGE: Final = "Reports"

RANDOM_FOREST_MODEL_NAME: Final = "random_forest"

MAXIMUM_SHARPE_SCENARIO: Final = "maximum_sharpe"

MINIMUM_VOLATILITY_SCENARIO: Final = "minimum_volatility"


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
    """Return readable scenario label."""
    try:
        return translate(f"decision.scenario.{scenario_name}", language)
    except KeyError:
        return scenario_name.replace("_", " ").title()


def _find_model(
    models: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any] | None:
    """Find forecast model by identifier."""
    for model in models:
        if model.get("model_name") == model_name:
            return model

    return None


def _find_scenario(
    scenarios: list[dict[str, Any]],
    scenario_name: str,
) -> dict[str, Any] | None:
    """Find scenario by identifier."""
    for scenario in scenarios:
        if scenario.get("name") == scenario_name:
            return scenario

    return None


def _load_dashboard_data(
    client: APIClient,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Load dashboard data from existing API endpoints."""
    forecast = client.get(FORECAST_SUMMARY_PATH)

    risk = client.get(RISK_SCENARIOS_PATH)

    portfolio = client.get(PORTFOLIO_SCENARIOS_PATH)

    decision = client.get(BALANCED_DECISION_PATH)

    rag_status = client.get(RAG_STATUS_PATH)

    agent_status = client.get(AGENT_STATUS_PATH)

    return (
        forecast,
        risk,
        portfolio,
        decision,
        rag_status,
        agent_status,
    )


def _build_executive_snapshot(
    decision: dict[str, Any],
    portfolio: dict[str, Any],
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """Combine deterministic recommendation with validation evidence."""
    recommended_name = decision.get("recommended_scenario")
    scenario_name = str(recommended_name) if recommended_name is not None else ""
    portfolio_scenario = _find_scenario(
        _as_mapping_list(portfolio.get("scenarios")),
        scenario_name,
    )
    validation = (
        _as_mapping(portfolio_scenario.get("validation"))
        if portfolio_scenario is not None
        else {}
    )
    recommended_decision = _find_scenario(
        _as_mapping_list(decision.get("scenarios")),
        scenario_name,
    )
    random_forest = _find_model(
        _as_mapping_list(forecast.get("models")),
        RANDOM_FOREST_MODEL_NAME,
    )
    forecast_metrics = (
        _as_mapping(random_forest.get("metrics")) if random_forest is not None else {}
    )

    return {
        "recommended_scenario": scenario_name or None,
        "admissible": (
            bool(recommended_decision.get("admissible"))
            if recommended_decision is not None
            else None
        ),
        "annualized_return": validation.get("annualized_mean_return"),
        "annualized_volatility": validation.get("annualized_volatility"),
        "sharpe_ratio": validation.get("sharpe_ratio"),
        "maximum_drawdown": validation.get("maximum_drawdown"),
        "directional_accuracy": forecast_metrics.get("directional_accuracy"),
        "evaluation_split": portfolio.get("evaluation_split", "validation"),
    }


def _render_executive_snapshot(
    decision: dict[str, Any],
    portfolio: dict[str, Any],
    forecast: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render an action-oriented balanced-profile decision summary."""
    snapshot = _build_executive_snapshot(
        decision,
        portfolio,
        forecast,
    )
    scenario_name = snapshot["recommended_scenario"]

    st.subheader(translate("dashboard.executive.heading", language))
    st.caption(translate("dashboard.executive.caption", language))

    if scenario_name is None:
        st.warning(translate("dashboard.executive.empty", language))
        return

    if snapshot["admissible"] is True:
        st.success(
            translate(
                "dashboard.executive.recommendation",
                language,
                scenario=_scenario_label(str(scenario_name), language),
            )
        )

    recommendation_column, return_column, volatility_column = st.columns(3)

    with recommendation_column:
        st.metric(
            translate("dashboard.executive.scenario", language),
            _scenario_label(str(scenario_name), language),
        )

    with return_column:
        st.metric(
            translate("dashboard.executive.return", language),
            _format_percentage(snapshot["annualized_return"]),
        )

    with volatility_column:
        st.metric(
            translate("dashboard.executive.volatility", language),
            _format_percentage(snapshot["annualized_volatility"]),
        )

    sharpe_column, drawdown_column, forecast_column = st.columns(3)

    with sharpe_column:
        st.metric(
            translate("dashboard.executive.sharpe", language),
            _format_decimal(snapshot["sharpe_ratio"]),
        )

    with drawdown_column:
        st.metric(
            translate("dashboard.executive.drawdown", language),
            _format_percentage(snapshot["maximum_drawdown"]),
        )

    with forecast_column:
        st.metric(
            translate("dashboard.executive.forecast", language),
            _format_percentage(snapshot["directional_accuracy"]),
        )

    st.caption(translate("dashboard.executive.validation_notice", language))


def _select_dashboard_destination(page_name: str) -> None:
    """Select an existing application module from a dashboard action."""
    st.session_state[SELECTED_PAGE_STATE_KEY] = page_name


def _render_quick_actions(
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render direct routes from summary to detailed analytical modules."""
    st.subheader(translate("dashboard.actions.heading", language))
    destinations = (
        (
            DECISION_WORKSPACE_PAGE,
            "dashboard.actions.decision",
            ":material/tune:",
        ),
        (
            FORECASTING_PAGE,
            "dashboard.actions.forecast",
            ":material/show_chart:",
        ),
        (
            MARKET_INTELLIGENCE_PAGE,
            "dashboard.actions.market",
            ":material/query_stats:",
        ),
        (
            REPORTS_PAGE,
            "dashboard.actions.reports",
            ":material/description:",
        ),
    )

    for column, (page_name, label_key, icon) in zip(
        st.columns(len(destinations)),
        destinations,
        strict=True,
    ):
        with column:
            st.button(
                translate(label_key, language),
                key=f"dashboard_action_{page_name.lower().replace(' ', '_')}",
                type=(
                    "primary" if page_name == DECISION_WORKSPACE_PAGE else "secondary"
                ),
                icon=icon,
                width="stretch",
                on_click=_select_dashboard_destination,
                args=(page_name,),
            )


def _render_forecast_kpi(
    forecast: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render Random Forest validation KPI."""
    models = _as_mapping_list(forecast.get("models"))

    random_forest = _find_model(
        models,
        RANDOM_FOREST_MODEL_NAME,
    )

    if random_forest is None:
        st.metric(
            translate("dashboard.kpi.direction", language),
            "—",
        )
        return

    metrics = _as_mapping(random_forest.get("metrics"))

    st.metric(
        translate("dashboard.kpi.direction", language),
        _format_percentage(metrics.get("directional_accuracy")),
    )


def _render_risk_kpis(
    risk: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render key validation risk KPIs."""
    scenarios = _as_mapping_list(risk.get("scenarios"))

    minimum_volatility = _find_scenario(
        scenarios,
        MINIMUM_VOLATILITY_SCENARIO,
    )

    maximum_sharpe = _find_scenario(
        scenarios,
        MAXIMUM_SHARPE_SCENARIO,
    )

    if maximum_sharpe is None:
        st.metric(
            translate("dashboard.kpi.max_sharpe", language),
            "—",
        )
    else:
        risk_block = _as_mapping(maximum_sharpe.get("risk"))

        metrics = _as_mapping(risk_block.get("metrics"))

        st.metric(
            translate("dashboard.kpi.max_sharpe", language),
            _format_decimal(metrics.get("sharpe_ratio")),
        )

    if minimum_volatility is None:
        st.metric(
            translate("dashboard.kpi.lowest_volatility", language),
            "—",
        )
    else:
        risk_block = _as_mapping(minimum_volatility.get("risk"))

        metrics = _as_mapping(risk_block.get("metrics"))

        st.metric(
            translate("dashboard.kpi.lowest_volatility", language),
            _format_percentage(metrics.get("annualized_volatility")),
        )


def _render_portfolio_kpi(
    portfolio: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render portfolio validation observation KPI."""
    scenarios = _as_mapping_list(portfolio.get("scenarios"))

    maximum_sharpe = _find_scenario(
        scenarios,
        MAXIMUM_SHARPE_SCENARIO,
    )

    if maximum_sharpe is None:
        st.metric(
            translate("dashboard.kpi.validation_observations", language),
            "—",
        )
        return

    validation = _as_mapping(maximum_sharpe.get("validation"))

    observations = validation.get("observations")

    st.metric(
        translate("dashboard.kpi.validation_observations", language),
        str(observations if observations is not None else "—"),
    )


def _render_main_kpis(
    forecast: dict[str, Any],
    risk: dict[str, Any],
    portfolio: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render primary analytical dashboard KPIs."""
    st.subheader(translate("dashboard.kpi.heading", language))

    (
        forecast_column,
        sharpe_column,
        volatility_column,
        observations_column,
    ) = st.columns(4)

    with forecast_column:
        _render_forecast_kpi(forecast, language)

    with sharpe_column:
        scenarios = _as_mapping_list(risk.get("scenarios"))

        maximum_sharpe = _find_scenario(
            scenarios,
            MAXIMUM_SHARPE_SCENARIO,
        )

        if maximum_sharpe is None:
            st.metric(
                translate("dashboard.kpi.max_sharpe", language),
                "—",
            )
        else:
            risk_block = _as_mapping(maximum_sharpe.get("risk"))

            metrics = _as_mapping(risk_block.get("metrics"))

            st.metric(
                translate("dashboard.kpi.max_sharpe", language),
                _format_decimal(metrics.get("sharpe_ratio")),
            )

    with volatility_column:
        scenarios = _as_mapping_list(risk.get("scenarios"))

        minimum_volatility = _find_scenario(
            scenarios,
            MINIMUM_VOLATILITY_SCENARIO,
        )

        if minimum_volatility is None:
            st.metric(
                translate("dashboard.kpi.lowest_volatility", language),
                "—",
            )
        else:
            risk_block = _as_mapping(minimum_volatility.get("risk"))

            metrics = _as_mapping(risk_block.get("metrics"))

            st.metric(
                translate("dashboard.kpi.lowest_volatility", language),
                _format_percentage(metrics.get("annualized_volatility")),
            )

    with observations_column:
        _render_portfolio_kpi(portfolio, language)


def _build_portfolio_snapshot(
    portfolio: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Build comparison of selected portfolio scenarios."""
    scenarios = _as_mapping_list(portfolio.get("scenarios"))

    rows: list[dict[str, object]] = []

    for scenario_name in [
        MAXIMUM_SHARPE_SCENARIO,
        MINIMUM_VOLATILITY_SCENARIO,
    ]:
        scenario = _find_scenario(
            scenarios,
            scenario_name,
        )

        if scenario is None:
            continue

        validation = _as_mapping(scenario.get("validation"))

        rows.append(
            {
                translate("dashboard.column.strategy", language): _scenario_label(
                    scenario_name, language
                ),
                translate("dashboard.column.return", language): (
                    _format_percentage(validation.get("annualized_mean_return"))
                ),
                translate("dashboard.column.volatility", language): (
                    _format_percentage(validation.get("annualized_volatility"))
                ),
                translate("dashboard.column.sharpe", language): _format_decimal(
                    validation.get("sharpe_ratio")
                ),
                translate("dashboard.column.drawdown", language): (
                    _format_percentage(validation.get("maximum_drawdown"))
                ),
                translate("dashboard.column.effective_assets", language): (
                    _format_decimal(validation.get("effective_number_of_assets"))
                ),
            }
        )

    return pd.DataFrame(rows)


def _render_portfolio_snapshot(
    portfolio: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render compact portfolio comparison."""
    st.subheader(translate("dashboard.portfolio.heading", language))

    frame = _build_portfolio_snapshot(portfolio, language)

    if frame.empty:
        st.warning(translate("dashboard.portfolio.empty", language))
        return

    render_centered_dataframe(
        frame,
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("dashboard.portfolio.caption", language))


def _render_system_capabilities(
    rag_status: dict[str, Any],
    agent_status: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render analytical module availability."""
    st.subheader(translate("dashboard.capabilities.heading", language))

    rows = [
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.rag_corpus", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.available", language)
                if bool(
                    rag_status.get(
                        "corpus_available",
                        False,
                    )
                )
                else translate("dashboard.status.missing", language)
            ),
        },
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.rag_benchmark", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.available", language)
                if bool(
                    rag_status.get(
                        "benchmark_available",
                        False,
                    )
                )
                else translate("dashboard.status.missing", language)
            ),
        },
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.forecast_agent", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.available", language)
                if bool(
                    agent_status.get(
                        "forecast_artifact_available",
                        False,
                    )
                )
                else translate("dashboard.status.missing", language)
            ),
        },
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.risk_agent", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.available", language)
                if bool(
                    agent_status.get(
                        "portfolio_artifact_available",
                        False,
                    )
                )
                else translate("dashboard.status.missing", language)
            ),
        },
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.allocation_agent", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.available", language)
                if bool(
                    agent_status.get(
                        "reinforcement_learning_artifact_available",
                        False,
                    )
                )
                else translate("dashboard.status.missing", language)
            ),
        },
        {
            translate("dashboard.column.component", language): translate(
                "dashboard.capability.agent_runtime", language
            ),
            translate("dashboard.column.status", language): (
                translate("dashboard.status.ready", language)
                if bool(
                    agent_status.get(
                        "runtime_inputs_available",
                        False,
                    )
                )
                else translate("dashboard.status.no_data", language)
            ),
        },
    ]

    render_centered_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


def _render_analytical_alerts(
    forecast: dict[str, Any],
    risk: dict[str, Any],
    agent_status: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render high-level analytical alerts."""
    st.subheader(translate("dashboard.alerts.heading", language))

    models = _as_mapping_list(forecast.get("models"))

    random_forest = _find_model(
        models,
        RANDOM_FOREST_MODEL_NAME,
    )

    if random_forest is not None:
        metrics = _as_mapping(random_forest.get("metrics"))

        directional_accuracy = _as_float(metrics.get("directional_accuracy"))

        r2 = _as_float(metrics.get("r2"))

        if 0.45 <= directional_accuracy <= 0.55:
            st.warning(translate("dashboard.alerts.direction", language))

        if abs(r2) < 0.05:
            st.warning(translate("dashboard.alerts.r2", language))

    scenarios = _as_mapping_list(risk.get("scenarios"))

    minimum_volatility = _find_scenario(
        scenarios,
        MINIMUM_VOLATILITY_SCENARIO,
    )

    if minimum_volatility is not None:
        risk_block = _as_mapping(minimum_volatility.get("risk"))

        metrics = _as_mapping(risk_block.get("metrics"))

        drawdown = _as_float(metrics.get("maximum_drawdown"))

        if drawdown > 0.10:
            st.info(translate("dashboard.alerts.drawdown", language))

    test_split_used = agent_status.get("test_split_used")

    if test_split_used is False:
        st.success(translate("dashboard.alerts.test_closed", language))


def render_dashboard_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render central analytical dashboard."""
    st.title(translate("dashboard.title", language))

    st.caption(translate("dashboard.caption", language))

    try:
        (
            forecast,
            risk,
            portfolio,
            decision,
            rag_status,
            agent_status,
        ) = _load_dashboard_data(client)
    except APIClientError as exc:
        st.error(translate("dashboard.error", language))

        st.caption(str(exc))

        return

    _render_executive_snapshot(
        decision,
        portfolio,
        forecast,
        language,
    )

    st.divider()

    _render_quick_actions(language)

    st.divider()

    _render_analytical_alerts(
        forecast,
        risk,
        agent_status,
        language,
    )

    st.divider()

    _render_portfolio_snapshot(portfolio, language)

    st.divider()

    with st.expander(translate("dashboard.technical.heading", language)):
        _render_system_capabilities(
            rag_status,
            agent_status,
            language,
        )

    with st.expander(translate("dashboard.scope.heading", language)):
        st.markdown(translate("dashboard.scope.body", language))

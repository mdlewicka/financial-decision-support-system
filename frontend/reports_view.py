"""Report generation view for the Streamlit frontend."""

import json
from datetime import UTC, datetime
from typing import Any, Final, cast

import streamlit as st

from frontend.api_client import (
    APIClient,
    APIClientError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate

FORECAST_SUMMARY_PATH: Final = "/api/v1/forecast/summary"
XAI_SUMMARY_PATH: Final = "/api/v1/xai/summary"
RISK_SCENARIOS_PATH: Final = "/api/v1/risk/scenarios"
PORTFOLIO_SCENARIOS_PATH: Final = "/api/v1/portfolio/scenarios"
RAG_STATUS_PATH: Final = "/api/v1/rag/status"
AGENT_STATUS_PATH: Final = "/api/v1/agents/status"

REPORT_DATA_STATE_KEY: Final = "reports_generated_data_v1"
REPORT_MARKDOWN_STATE_KEY: Final = "reports_generated_markdown_v1"

REPORT_FILENAME_MARKDOWN: Final = "financial_decision_support_report.md"
REPORT_FILENAME_JSON: Final = "financial_decision_support_report.json"

RANDOM_FOREST_MODEL_NAME: Final = "random_forest"


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
) -> float | None:
    """Convert a JSON value to float."""
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None


def _format_decimal(
    value: Any,
    *,
    digits: int = 3,
) -> str:
    """Format a decimal metric."""
    number = _as_float(value)

    if number is None:
        return "—"

    return f"{number:.{digits}f}"


def _format_percentage(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    """Format a decimal value as percentage."""
    number = _as_float(value)

    if number is None:
        return "—"

    return f"{number:.{digits}%}"


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
    """Find model by identifier."""
    for model in models:
        if model.get("model_name") == model_name:
            return model

    return None


def _load_report_sources(
    client: APIClient,
) -> dict[str, dict[str, Any]]:
    """Load all source data required by the report."""
    return {
        "system": (client.get_system_info()),
        "forecast": client.get(FORECAST_SUMMARY_PATH),
        "xai": client.get(XAI_SUMMARY_PATH),
        "risk": client.get(RISK_SCENARIOS_PATH),
        "portfolio": client.get(PORTFOLIO_SCENARIOS_PATH),
        "rag": client.get(RAG_STATUS_PATH),
        "agents": client.get(AGENT_STATUS_PATH),
    }


def _build_forecast_section(
    forecast: dict[str, Any],
) -> dict[str, Any]:
    """Build compact forecasting report section."""
    models = _as_mapping_list(forecast.get("models"))

    random_forest = _find_model(
        models,
        RANDOM_FOREST_MODEL_NAME,
    )

    random_forest_metrics: dict[str, Any] = {}

    if random_forest is not None:
        random_forest_metrics = _as_mapping(random_forest.get("metrics"))

    return {
        "evaluation_split": (forecast.get("evaluation_split")),
        "target_definition": (forecast.get("target_definition")),
        "tickers": forecast.get(
            "tickers",
            [],
        ),
        "random_forest": {
            "observations": (random_forest_metrics.get("observations")),
            "mae": (random_forest_metrics.get("mae")),
            "rmse": (random_forest_metrics.get("rmse")),
            "r2": (random_forest_metrics.get("r2")),
            "directional_accuracy": (random_forest_metrics.get("directional_accuracy")),
        },
    }


def _build_xai_section(
    xai: dict[str, Any],
) -> dict[str, Any]:
    """Build explainability report section."""
    features = _as_mapping_list(xai.get("features"))

    top_features: list[dict[str, Any]] = []

    for feature in features[:5]:
        top_features.append(
            {
                "feature": feature.get("feature"),
                "mean_absolute_shap_value": (feature.get("mean_absolute_shap_value")),
            }
        )

    return {
        "model_name": xai.get("model_name"),
        "explanation_split": xai.get("explanation_split"),
        "target_definition": xai.get("target_definition"),
        "methods": xai.get(
            "methods",
            [],
        ),
        "global_observations": xai.get("global_observations"),
        "top_features": top_features,
    }


def _build_risk_section(
    risk: dict[str, Any],
) -> dict[str, Any]:
    """Build risk report section."""
    scenarios = _as_mapping_list(risk.get("scenarios"))

    result: list[dict[str, Any]] = []

    for scenario in scenarios:
        risk_block = _as_mapping(scenario.get("risk"))

        metrics = _as_mapping(risk_block.get("metrics"))

        result.append(
            {
                "name": scenario.get("name"),
                "annualized_return": (metrics.get("annualized_mean_return")),
                "annualized_volatility": (metrics.get("annualized_volatility")),
                "sharpe_ratio": (metrics.get("sharpe_ratio")),
                "value_at_risk": (metrics.get("value_at_risk")),
                "conditional_value_at_risk": (metrics.get("conditional_value_at_risk")),
                "maximum_drawdown": (metrics.get("maximum_drawdown")),
                "concentration_index": (risk_block.get("concentration_index")),
                "effective_number_of_assets": (
                    risk_block.get("effective_number_of_assets")
                ),
            }
        )

    return {
        "evaluation_split": risk.get("evaluation_split"),
        "scenarios": result,
    }


def _build_portfolio_section(
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    """Build portfolio report section."""
    scenarios = _as_mapping_list(portfolio.get("scenarios"))

    result: list[dict[str, Any]] = []

    for scenario in scenarios:
        validation = _as_mapping(scenario.get("validation"))

        weights = _as_mapping_list(scenario.get("weights"))

        result.append(
            {
                "name": scenario.get("name"),
                "construction": (scenario.get("construction")),
                "weights": weights,
                "validation": {
                    "observations": (validation.get("observations")),
                    "annualized_return": (validation.get("annualized_mean_return")),
                    "annualized_volatility": (validation.get("annualized_volatility")),
                    "sharpe_ratio": (validation.get("sharpe_ratio")),
                    "maximum_drawdown": (validation.get("maximum_drawdown")),
                    "concentration_index": (validation.get("concentration_index")),
                    "effective_number_of_assets": (
                        validation.get("effective_number_of_assets")
                    ),
                },
            }
        )

    return {
        "evaluation_split": (portfolio.get("evaluation_split")),
        "tickers": portfolio.get(
            "tickers",
            [],
        ),
        "scenarios": result,
    }


def _build_report_data(
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build reproducible structured report data."""
    agents = sources["agents"]

    return {
        "metadata": {
            "title": ("Financial Decision Support System Report"),
            "generated_at_utc": (datetime.now(UTC).isoformat()),
            "evaluation_scope": ("validation"),
            "test_split_used": (agents.get("test_split_used")),
        },
        "system": sources["system"],
        "forecasting": (_build_forecast_section(sources["forecast"])),
        "explainability": (_build_xai_section(sources["xai"])),
        "risk": (_build_risk_section(sources["risk"])),
        "portfolio": (_build_portfolio_section(sources["portfolio"])),
        "rag": {
            "corpus_available": (sources["rag"].get("corpus_available")),
            "corpus_documents": (sources["rag"].get("corpus_documents")),
            "benchmark_available": (sources["rag"].get("benchmark_available")),
            "model_name": (sources["rag"].get("model_name")),
        },
        "agents": {
            "runtime_inputs_available": (agents.get("runtime_inputs_available")),
            "model_name": (agents.get("model_name")),
            "retrieval_strategy": (agents.get("retrieval_strategy")),
            "evaluation_basis": (agents.get("evaluation_basis")),
            "test_split_used": (agents.get("test_split_used")),
        },
    }


def _append_forecast_markdown(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    """Append forecasting section to Markdown report."""
    forecasting = _as_mapping(report.get("forecasting"))

    random_forest = _as_mapping(forecasting.get("random_forest"))

    lines.extend(
        [
            "## Forecasting",
            "",
            (f"- Zbiór ewaluacyjny: {forecasting.get('evaluation_split', '—')}"),
            (f"- Definicja celu: {forecasting.get('target_definition', '—')}"),
            (
                "- Random Forest – liczba obserwacji: "
                f"{random_forest.get('observations', '—')}"
            ),
            (
                "- Random Forest – MAE: "
                f"{_format_decimal(random_forest.get('mae'), digits=6)}"
            ),
            (
                "- Random Forest – RMSE: "
                f"{_format_decimal(random_forest.get('rmse'), digits=6)}"
            ),
            (
                "- Random Forest – R²: "
                f"{_format_decimal(random_forest.get('r2'), digits=4)}"
            ),
            (
                "- Random Forest – trafność kierunku: "
                f"{_format_percentage(random_forest.get('directional_accuracy'))}"
            ),
            "",
        ]
    )


def _append_xai_markdown(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    """Append explainability section."""
    explainability = _as_mapping(report.get("explainability"))

    lines.extend(
        [
            "## Explainability and Governance",
            "",
            (f"- Model: {explainability.get('model_name', '—')}"),
            (f"- Zbiór wyjaśnień: {explainability.get('explanation_split', '—')}"),
            (
                "- Liczba obserwacji globalnych: "
                f"{explainability.get('global_observations', '—')}"
            ),
            "",
            "### Najważniejsze cechy według SHAP",
            "",
        ]
    )

    features = _as_mapping_list(explainability.get("top_features"))

    if not features:
        lines.extend(
            [
                "Brak danych.",
                "",
            ]
        )
        return

    for index, feature in enumerate(
        features,
        start=1,
    ):
        lines.append(
            f"{index}. "
            f"{feature.get('feature', '—')} – "
            f"{_format_decimal(feature.get('mean_absolute_shap_value'), digits=6)}"
        )

    lines.append("")


def _append_risk_markdown(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    """Append risk section."""
    risk = _as_mapping(report.get("risk"))

    scenarios = _as_mapping_list(risk.get("scenarios"))

    lines.extend(
        [
            "## Risk and Scenarios",
            "",
            (f"- Zbiór ewaluacyjny: {risk.get('evaluation_split', '—')}"),
            "",
        ]
    )

    for scenario in scenarios:
        scenario_name = str(
            scenario.get(
                "name",
                "—",
            )
        )

        lines.extend(
            [
                (f"### {_scenario_label(scenario_name)}"),
                "",
                (
                    "- Zwrot roczny: "
                    f"{_format_percentage(scenario.get('annualized_return'))}"
                ),
                (
                    "- Zmienność roczna: "
                    f"{_format_percentage(scenario.get('annualized_volatility'))}"
                ),
                (f"- Sharpe: {_format_decimal(scenario.get('sharpe_ratio'))}"),
                (f"- VaR: {_format_percentage(scenario.get('value_at_risk'))}"),
                (
                    "- CVaR: "
                    f"{_format_percentage(scenario.get('conditional_value_at_risk'))}"
                ),
                (
                    "- Maksymalny drawdown: "
                    f"{_format_percentage(scenario.get('maximum_drawdown'))}"
                ),
                "",
            ]
        )


def _append_portfolio_markdown(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    """Append portfolio section."""
    portfolio = _as_mapping(report.get("portfolio"))

    scenarios = _as_mapping_list(portfolio.get("scenarios"))

    lines.extend(
        [
            "## Optimization",
            "",
            (f"- Zbiór ewaluacyjny: {portfolio.get('evaluation_split', '—')}"),
            "",
        ]
    )

    for scenario in scenarios:
        scenario_name = str(
            scenario.get(
                "name",
                "—",
            )
        )

        validation = _as_mapping(scenario.get("validation"))

        lines.extend(
            [
                (f"### {_scenario_label(scenario_name)}"),
                "",
                (
                    "- Zwrot roczny: "
                    f"{_format_percentage(validation.get('annualized_return'))}"
                ),
                (
                    "- Zmienność roczna: "
                    f"{_format_percentage(validation.get('annualized_volatility'))}"
                ),
                (f"- Sharpe: {_format_decimal(validation.get('sharpe_ratio'))}"),
                (
                    "- Maksymalny drawdown: "
                    f"{_format_percentage(validation.get('maximum_drawdown'))}"
                ),
                "",
                "#### Wagi",
                "",
            ]
        )

        weights = _as_mapping_list(scenario.get("weights"))

        if not weights:
            lines.append("Brak danych.")

        for weight in weights:
            lines.append(
                "- "
                f"{weight.get('ticker', '—')}: "
                f"{_format_percentage(weight.get('weight'))}"
            )

        lines.append("")


def _append_ai_markdown(
    lines: list[str],
    report: dict[str, Any],
) -> None:
    """Append RAG and agent metadata."""
    rag = _as_mapping(report.get("rag"))

    agents = _as_mapping(report.get("agents"))

    lines.extend(
        [
            "## AI Insights",
            "",
            (f"- Korpus RAG dostępny: {rag.get('corpus_available', '—')}"),
            (f"- Liczba dokumentów korpusu: {rag.get('corpus_documents', '—')}"),
            (f"- Benchmark RAG dostępny: {rag.get('benchmark_available', '—')}"),
            (f"- Model RAG: {rag.get('model_name', '—')}"),
            (
                "- Runtime agentów gotowy: "
                f"{agents.get('runtime_inputs_available', '—')}"
            ),
            (f"- Model agentów: {agents.get('model_name', '—')}"),
            (f"- Strategia retrieval: {agents.get('retrieval_strategy', '—')}"),
            "",
        ]
    )


def _build_markdown_report(
    report: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Render structured report as Markdown."""
    if language == "en":
        return _build_english_markdown_report(report)

    metadata = _as_mapping(report.get("metadata"))

    system = _as_mapping(report.get("system"))

    lines = [
        "# Financial Decision Support System",
        "",
        "## Raport analityczny",
        "",
        (f"- Wygenerowano UTC: {metadata.get('generated_at_utc', '—')}"),
        (f"- Zakres ewaluacji: {metadata.get('evaluation_scope', '—')}"),
        (f"- Final test użyty: {metadata.get('test_split_used', '—')}"),
        (f"- API: {system.get('title', '—')}"),
        (f"- Wersja API: {system.get('version', '—')}"),
        "",
    ]

    _append_forecast_markdown(
        lines,
        report,
    )

    _append_xai_markdown(
        lines,
        report,
    )

    _append_risk_markdown(
        lines,
        report,
    )

    _append_portfolio_markdown(
        lines,
        report,
    )

    _append_ai_markdown(
        lines,
        report,
    )

    lines.extend(
        [
            "## Ograniczenia",
            "",
            ("- Raport agreguje wyniki z istniejących artefaktów i endpointów API."),
            ("- Frontend nie wykonuje ponownego trenowania modeli ani optymalizacji."),
            ("- Wyniki dotyczą zbioru walidacyjnego."),
            (
                "- Końcowy zbiór testowy pozostaje "
                "zamknięty do etapu finalnej ewaluacji."
            ),
            (
                "- Wyniki modeli prognostycznych, "
                "analizy ryzyka i optymalizacji "
                "powinny być interpretowane łącznie."
            ),
            ("- Raport nie stanowi indywidualnej rekomendacji inwestycyjnej."),
            "",
        ]
    )

    return "\n".join(lines)


def _build_english_markdown_report(
    report: dict[str, Any],
) -> str:
    """Render the same analytical evidence as an English Markdown report."""
    metadata = _as_mapping(report.get("metadata"))
    system = _as_mapping(report.get("system"))
    forecasting = _as_mapping(report.get("forecasting"))
    random_forest = _as_mapping(forecasting.get("random_forest"))
    explainability = _as_mapping(report.get("explainability"))
    risk = _as_mapping(report.get("risk"))
    portfolio = _as_mapping(report.get("portfolio"))
    rag = _as_mapping(report.get("rag"))
    agents = _as_mapping(report.get("agents"))

    lines = [
        "# Financial Decision Support System",
        "",
        "## Analytical report",
        "",
        f"- Generated at UTC: {metadata.get('generated_at_utc', '—')}",
        f"- Evaluation scope: {metadata.get('evaluation_scope', '—')}",
        f"- Final test used: {metadata.get('test_split_used', '—')}",
        f"- API: {system.get('title', '—')}",
        f"- API version: {system.get('version', '—')}",
        "",
        "## Forecasting",
        "",
        f"- Evaluation split: {forecasting.get('evaluation_split', '—')}",
        f"- Target definition: {forecasting.get('target_definition', '—')}",
        f"- Random Forest observations: {random_forest.get('observations', '—')}",
        f"- Random Forest MAE: {_format_decimal(random_forest.get('mae'), digits=6)}",
        f"- Random Forest RMSE: {_format_decimal(random_forest.get('rmse'), digits=6)}",
        f"- Random Forest R²: {_format_decimal(random_forest.get('r2'), digits=4)}",
        (
            "- Random Forest directional accuracy: "
            f"{_format_percentage(random_forest.get('directional_accuracy'))}"
        ),
        "",
        "## Explainability and Governance",
        "",
        f"- Model: {explainability.get('model_name', '—')}",
        f"- Explanation split: {explainability.get('explanation_split', '—')}",
        (f"- Global observations: {explainability.get('global_observations', '—')}"),
        "",
        "### Top SHAP features",
        "",
    ]

    features = _as_mapping_list(explainability.get("top_features"))

    if features:
        for index, feature in enumerate(features, start=1):
            lines.append(
                f"{index}. {feature.get('feature', '—')} – "
                f"{_format_decimal(feature.get('mean_absolute_shap_value'), digits=6)}"
            )
    else:
        lines.append("No data.")

    lines.extend(
        [
            "",
            "## Risk and Scenarios",
            "",
            f"- Evaluation split: {risk.get('evaluation_split', '—')}",
            "",
        ]
    )

    for scenario in _as_mapping_list(risk.get("scenarios")):
        scenario_name = str(scenario.get("name", "—"))
        lines.extend(
            [
                f"### {_scenario_label(scenario_name, 'en')}",
                "",
                (
                    "- Annualized return: "
                    f"{_format_percentage(scenario.get('annualized_return'))}"
                ),
                (
                    "- Annualized volatility: "
                    f"{_format_percentage(scenario.get('annualized_volatility'))}"
                ),
                f"- Sharpe: {_format_decimal(scenario.get('sharpe_ratio'))}",
                f"- VaR: {_format_percentage(scenario.get('value_at_risk'))}",
                (
                    "- CVaR: "
                    f"{_format_percentage(scenario.get('conditional_value_at_risk'))}"
                ),
                (
                    "- Maximum drawdown: "
                    f"{_format_percentage(scenario.get('maximum_drawdown'))}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Optimization",
            "",
            f"- Evaluation split: {portfolio.get('evaluation_split', '—')}",
            "",
        ]
    )

    for scenario in _as_mapping_list(portfolio.get("scenarios")):
        scenario_name = str(scenario.get("name", "—"))
        validation = _as_mapping(scenario.get("validation"))
        weights = _as_mapping_list(scenario.get("weights"))
        lines.extend(
            [
                f"### {_scenario_label(scenario_name, 'en')}",
                "",
                (
                    "- Annualized return: "
                    f"{_format_percentage(validation.get('annualized_return'))}"
                ),
                (
                    "- Annualized volatility: "
                    f"{_format_percentage(validation.get('annualized_volatility'))}"
                ),
                f"- Sharpe: {_format_decimal(validation.get('sharpe_ratio'))}",
                (
                    "- Maximum drawdown: "
                    f"{_format_percentage(validation.get('maximum_drawdown'))}"
                ),
                "",
                "#### Weights",
                "",
            ]
        )

        if weights:
            for weight in weights:
                lines.append(
                    f"- {weight.get('ticker', '—')}: "
                    f"{_format_percentage(weight.get('weight'))}"
                )
        else:
            lines.append("No data.")

        lines.append("")

    lines.extend(
        [
            "## AI Insights",
            "",
            f"- RAG corpus available: {rag.get('corpus_available', '—')}",
            f"- Corpus documents: {rag.get('corpus_documents', '—')}",
            f"- RAG benchmark available: {rag.get('benchmark_available', '—')}",
            f"- RAG model: {rag.get('model_name', '—')}",
            f"- Agent runtime ready: {agents.get('runtime_inputs_available', '—')}",
            f"- Agent model: {agents.get('model_name', '—')}",
            f"- Retrieval strategy: {agents.get('retrieval_strategy', '—')}",
            "",
            "## Limitations",
            "",
            "- The report aggregates existing artifacts and API endpoint results.",
            "- The frontend does not retrain models or rerun optimization.",
            "- Results concern the validation split.",
            "- The final test split remains closed until final evaluation.",
            (
                "- Forecast, risk and optimization results should be interpreted "
                "together."
            ),
            "- The report does not constitute individualized investment advice.",
            "",
        ]
    )

    return "\n".join(lines)


def _render_report_summary(
    report: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render compact metadata above report preview."""
    metadata = _as_mapping(report.get("metadata"))

    agents = _as_mapping(report.get("agents"))

    (
        scope_column,
        test_column,
        model_column,
    ) = st.columns(3)

    with scope_column:
        st.metric(
            translate("reports.scope", language),
            str(
                metadata.get(
                    "evaluation_scope",
                    "—",
                )
            ),
        )

    with test_column:
        test_split_used = metadata.get("test_split_used")

        st.metric(
            translate("reports.final_test", language),
            (
                translate("ai.used", language)
                if test_split_used is True
                else translate("ai.closed", language)
                if test_split_used is False
                else "—"
            ),
        )

    with model_column:
        st.metric(
            translate("reports.ai_model", language),
            str(
                agents.get(
                    "model_name",
                    "—",
                )
            ),
        )


def _render_downloads(
    report: dict[str, Any],
    markdown_report: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render report download controls."""
    json_report = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    )

    markdown_column, json_column = st.columns(2)

    with markdown_column:
        st.download_button(
            translate("reports.download_markdown", language),
            data=markdown_report,
            file_name=(REPORT_FILENAME_MARKDOWN),
            mime="text/markdown",
            width="stretch",
        )

    with json_column:
        st.download_button(
            translate("reports.download_json", language),
            data=json_report,
            file_name=(REPORT_FILENAME_JSON),
            mime="application/json",
            width="stretch",
        )


def render_reports_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render analytical report generation page."""
    st.title(translate("reports.title", language))

    st.caption(translate("reports.caption", language))

    st.info(translate("reports.info", language))

    generate_report = st.button(
        translate("reports.generate", language),
        type="primary",
        key="reports_generate_v1",
    )

    if generate_report:
        try:
            with st.spinner(translate("reports.spinner", language)):
                sources = _load_report_sources(client)

                report = _build_report_data(sources)

                markdown_report = _build_markdown_report(report, language)
        except APIClientError as exc:
            st.error(translate("reports.error", language))

            st.caption(str(exc))
        else:
            st.session_state[REPORT_DATA_STATE_KEY] = report

            st.session_state[REPORT_MARKDOWN_STATE_KEY] = markdown_report

    report_value = st.session_state.get(REPORT_DATA_STATE_KEY)

    if not isinstance(report_value, dict):
        st.caption(translate("reports.empty", language))
        return

    report = cast(
        dict[str, Any],
        report_value,
    )

    markdown_value = _build_markdown_report(report, language)
    st.session_state[REPORT_MARKDOWN_STATE_KEY] = markdown_value

    st.divider()

    st.subheader(translate("reports.summary", language))

    _render_report_summary(report, language)

    st.subheader(translate("reports.export", language))

    _render_downloads(
        report,
        markdown_value,
        language,
    )

    st.subheader(translate("reports.preview", language))

    with st.expander(
        translate("reports.preview_markdown", language),
        expanded=False,
    ):
        st.markdown(markdown_value)

    st.warning(translate("reports.warning", language))

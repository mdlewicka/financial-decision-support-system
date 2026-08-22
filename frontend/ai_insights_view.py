"""AI Insights view combining RAG and decision-support agents."""

from typing import Any, Final, cast

import pandas as pd
import streamlit as st

from frontend.api_client import (
    APIClient,
    APIClientError,
    APITimeoutError,
)
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import render_centered_dataframe

RAG_STATUS_PATH: Final = "/api/v1/rag/status"
RAG_BENCHMARK_PATH: Final = "/api/v1/rag/benchmark"
RAG_QUERY_PATH: Final = "/api/v1/rag/query"

AGENT_STATUS_PATH: Final = "/api/v1/agents/status"
AGENT_RESULTS_PATH: Final = "/api/v1/agents/results"
AGENT_FORECAST_PATH: Final = "/api/v1/agents/forecast"
AGENT_RISK_PATH: Final = "/api/v1/agents/risk"
AGENT_ALLOCATION_PATH: Final = "/api/v1/agents/allocation"

GENERATIVE_TIMEOUT_SECONDS: Final = 60.0

RAG_RESPONSE_STATE_KEY: Final = "ai_insights_rag_response_v2"

AGENT_RESPONSE_STATE_KEY: Final = "ai_insights_agent_response_v2"


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
    """Normalize a JSON sequence of strings."""
    if not isinstance(
        value,
        (
            list,
            tuple,
        ),
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


def _strategy_label(
    strategy: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return readable retrieval strategy label."""
    try:
        return translate(f"ai.strategy.{strategy}", language)
    except KeyError:
        return strategy.replace("_", " ").title()


def _agent_type_label(
    agent_type: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return readable agent type label."""
    try:
        return translate(f"ai.agent_type.{agent_type}", language)
    except KeyError:
        return agent_type


def _agent_name_label(
    agent_name: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return readable agent name."""
    try:
        return translate(f"ai.agent_name.{agent_name}", language)
    except KeyError:
        return agent_name.replace("_", " ").title()


def _risk_preference_label(
    preference: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return readable risk preference label."""
    try:
        return translate(f"ai.risk.{preference}", language)
    except KeyError:
        return preference


def _clear_rag_response() -> None:
    """Remove stale RAG output after input changes."""
    st.session_state.pop(
        RAG_RESPONSE_STATE_KEY,
        None,
    )


def _clear_agent_response() -> None:
    """Remove stale agent output after input changes."""
    st.session_state.pop(
        AGENT_RESPONSE_STATE_KEY,
        None,
    )


def _render_timeout_message(
    *,
    operation: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render a clear message for long generative requests."""
    st.error(translate("ai.timeout", language, operation=operation))
    st.caption(
        translate(
            "ai.timeout.help",
            language,
            seconds=f"{GENERATIVE_TIMEOUT_SECONDS:g}",
        )
    )


def _render_rag_status(
    status: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render RAG availability metadata."""
    (
        corpus_column,
        evaluation_column,
        benchmark_column,
        model_column,
    ) = st.columns(4)

    with corpus_column:
        st.metric(
            translate("ai.rag.metric.corpus", language),
            (
                translate("ai.available", language)
                if bool(
                    status.get(
                        "corpus_available",
                        False,
                    )
                )
                else translate("ai.missing", language)
            ),
        )

    with evaluation_column:
        st.metric(
            translate("ai.rag.metric.evaluation", language),
            (
                translate("ai.available", language)
                if bool(
                    status.get(
                        "golden_dataset_available",
                        False,
                    )
                )
                else translate("ai.missing", language)
            ),
        )

    with benchmark_column:
        st.metric(
            translate("ai.rag.metric.benchmark", language),
            (
                translate("ai.available", language)
                if bool(
                    status.get(
                        "benchmark_available",
                        False,
                    )
                )
                else translate("ai.missing", language)
            ),
        )

    with model_column:
        st.metric(
            translate("ai.rag.metric.model", language),
            str(
                status.get(
                    "model_name",
                    "—",
                )
            ),
        )

    details: list[str] = []

    corpus_documents = status.get("corpus_documents")

    evaluation_items = status.get("golden_dataset_items")

    if corpus_documents is not None:
        details.append(translate("ai.rag.documents", language, count=corpus_documents))

    if evaluation_items is not None:
        details.append(translate("ai.rag.queries", language, count=evaluation_items))

    if details:
        st.caption(" · ".join(details))


def _render_rag_citations(
    citations: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render evidence returned by RAG."""
    if not citations:
        st.info(translate("ai.rag.citations_empty", language))
        return

    rows: list[dict[str, object]] = []

    for citation in citations:
        page = citation.get("page")

        rows.append(
            {
                translate("ai.rag.column.document", language): str(
                    citation.get(
                        "document_id",
                        "—",
                    )
                ),
                translate("ai.rag.column.title", language): str(
                    citation.get(
                        "title",
                        "—",
                    )
                ),
                translate("ai.rag.column.source", language): str(
                    citation.get(
                        "source",
                        "—",
                    )
                ),
                translate("ai.rag.column.page", language): (
                    page if page is not None else "—"
                ),
            }
        )

    render_centered_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


def _render_rag_response(
    response: dict[str, Any],
    selected_strategy: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render generated RAG answer."""
    st.divider()

    st.subheader(translate("ai.rag.answer", language))

    st.write(
        str(
            response.get(
                "answer",
                translate("ai.rag.answer_empty", language),
            )
        )
    )

    response_strategy = str(
        response.get(
            "retrieval_strategy",
            response.get(
                "strategy",
                selected_strategy,
            ),
        )
    )

    metadata = [
        translate(
            "ai.rag.response_strategy",
            language,
            strategy=_strategy_label(response_strategy, language),
        )
    ]

    retrieved_chunks = response.get("retrieved_chunk_count")

    if retrieved_chunks is not None:
        metadata.append(
            translate("ai.rag.chunk_count", language, count=retrieved_chunks)
        )

    st.caption(" · ".join(metadata))

    st.subheader(translate("ai.rag.sources", language))

    _render_rag_citations(_as_mapping_list(response.get("citations")), language)

    st.warning(translate("ai.rag.warning", language))


def _render_rag_benchmark(
    benchmark: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render retrieval benchmark."""
    best_configuration = str(
        benchmark.get(
            "best_configuration",
            "—",
        )
    )

    st.caption(
        translate(
            "ai.rag.benchmark_current",
            language,
            strategy=_strategy_label(best_configuration, language),
        )
    )

    configurations = _as_mapping_list(benchmark.get("configurations"))

    if not configurations:
        configurations = _as_mapping_list(benchmark.get("results"))

    if not configurations:
        st.info(translate("ai.rag.benchmark_empty", language))
        return

    rows: list[dict[str, object]] = []

    for configuration in configurations:
        strategy = str(
            configuration.get(
                "strategy",
                configuration.get(
                    "configuration",
                    "",
                ),
            )
        )

        rows.append(
            {
                translate("ai.rag.column.strategy", language): _strategy_label(
                    strategy, language
                ),
                "Quality score": (configuration.get("quality_score")),
                "Recall@k": (configuration.get("recall_at_k")),
                "Precision@k": (configuration.get("precision_at_k")),
                "MRR": (configuration.get("mean_reciprocal_rank")),
                translate("ai.rag.column.duration", language): configuration.get(
                    "duration_seconds"
                ),
            }
        )

    render_centered_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )

    st.caption(translate("ai.rag.benchmark_note", language))


def _render_rag_tab(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render the source-grounded RAG interface."""
    st.subheader(translate("ai.rag.heading", language))

    st.caption(translate("ai.rag.caption", language))

    try:
        status = client.get(RAG_STATUS_PATH)
    except APIClientError as exc:
        st.error(translate("ai.rag.status_error", language))

        st.caption(str(exc))
        return

    _render_rag_status(status, language)

    strategies = _as_string_list(status.get("strategies"))

    if not strategies:
        strategies = [
            "vector",
            "hybrid",
            "reranked",
        ]

    default_strategy = str(
        status.get(
            "default_strategy",
            "vector",
        )
    )

    if default_strategy not in strategies:
        default_strategy = strategies[0]

    st.divider()

    question = st.text_area(
        translate("ai.rag.question", language),
        value="",
        placeholder=translate("ai.rag.placeholder", language),
        height=105,
        key="ai_insights_rag_question_v2",
        on_change=_clear_rag_response,
    )

    selected_strategy = str(
        st.selectbox(
            translate("ai.rag.strategy", language),
            strategies,
            index=strategies.index(default_strategy),
            format_func=lambda value: _strategy_label(value, language),
            key="ai_insights_rag_strategy_v2",
            on_change=_clear_rag_response,
        )
    )

    submitted = st.button(
        translate("ai.rag.submit", language),
        type="primary",
        key="ai_insights_rag_submit_v2",
    )

    if submitted:
        normalized_question = question.strip()

        if not normalized_question:
            st.warning(translate("ai.rag.question_empty", language))
        else:
            try:
                with st.spinner(translate("ai.rag.spinner", language)):
                    response = client.post(
                        RAG_QUERY_PATH,
                        {
                            "question": (normalized_question),
                            "strategy": (selected_strategy),
                        },
                        timeout_seconds=(GENERATIVE_TIMEOUT_SECONDS),
                    )
            except APITimeoutError:
                _render_timeout_message(
                    operation=translate("ai.rag.operation", language),
                    language=language,
                )
            except APIClientError as exc:
                st.error(translate("ai.rag.error", language))

                st.caption(str(exc))
            else:
                st.session_state[RAG_RESPONSE_STATE_KEY] = response

    response_value = st.session_state.get(RAG_RESPONSE_STATE_KEY)

    if isinstance(
        response_value,
        dict,
    ):
        response = cast(
            dict[str, Any],
            response_value,
        )

        response_question = str(
            response.get(
                "question",
                "",
            )
        ).strip()

        if question.strip() and response_question == question.strip():
            _render_rag_response(
                response,
                selected_strategy,
                language,
            )

    st.divider()

    try:
        benchmark = client.get(RAG_BENCHMARK_PATH)
    except APIClientError as exc:
        with st.expander(translate("ai.rag.benchmark", language)):
            st.warning(translate("ai.rag.benchmark_unavailable", language))

            st.caption(str(exc))
    else:
        with st.expander(translate("ai.rag.benchmark", language)):
            _render_rag_benchmark(benchmark, language)

    with st.expander(translate("ai.rag.limitations", language)):
        st.markdown(translate("ai.rag.limitations_body", language))


def _render_agent_status(
    status: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render decision-agent runtime metadata."""
    (
        runtime_column,
        model_column,
        retrieval_column,
        test_column,
    ) = st.columns(4)

    with runtime_column:
        st.metric(
            translate("ai.agents.metric.runtime", language),
            (
                translate("ai.ready", language)
                if bool(
                    status.get(
                        "runtime_inputs_available",
                        False,
                    )
                )
                else translate("ai.no_data", language)
            ),
        )

    with model_column:
        st.metric(
            translate("ai.agents.metric.model", language),
            str(
                status.get(
                    "model_name",
                    "—",
                )
            ),
        )

    retrieval_strategy = status.get("retrieval_strategy")

    with retrieval_column:
        st.metric(
            translate("ai.agents.metric.retrieval", language),
            (
                _strategy_label(str(retrieval_strategy), language)
                if retrieval_strategy is not None
                else "—"
            ),
        )

    test_split_used = status.get("test_split_used")

    with test_column:
        st.metric(
            translate("ai.agents.metric.test", language),
            (
                translate("ai.used", language)
                if test_split_used is True
                else translate("ai.closed", language)
                if test_split_used is False
                else "—"
            ),
        )

    evaluation_basis = status.get("evaluation_basis")

    if evaluation_basis is not None:
        st.caption(
            translate("ai.agents.evaluation_basis", language, basis=evaluation_basis)
        )

    if test_split_used is False:
        st.success(translate("ai.agents.validation_ok", language))
    elif test_split_used is True:
        st.warning(translate("ai.agents.test_warning", language))


def _render_text_items(
    title: str,
    items: list[str],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render structured agent text items."""
    st.markdown(f"#### {title}")

    if not items:
        st.write(translate("ai.agents.no_items", language))
        return

    for item in items:
        st.markdown(f"- {item}")


def _render_agent_evidence(
    evidence: list[dict[str, Any]],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render agent evidence references."""
    if not evidence:
        st.info(translate("ai.agents.evidence_empty", language))
        return

    rows: list[dict[str, object]] = []

    for item in evidence:
        rows.append(
            {
                translate("ai.agents.column.source_type", language): str(
                    item.get(
                        "source_type",
                        "—",
                    )
                ),
                translate("ai.agents.column.identifier", language): str(
                    item.get(
                        "identifier",
                        "—",
                    )
                ),
                translate("ai.agents.column.description", language): str(
                    item.get(
                        "description",
                        "—",
                    )
                ),
            }
        )

    render_centered_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


def _render_agent_response(
    response: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render one structured agent response."""
    st.divider()

    agent_name = str(
        response.get(
            "agent_name",
            "—",
        )
    )

    st.subheader(_agent_name_label(agent_name, language))

    question = str(
        response.get(
            "question",
            "",
        )
    )

    if question:
        st.caption(translate("ai.agents.question_label", language, question=question))

    st.markdown(f"### {translate('ai.agents.summary', language)}")

    st.write(
        str(
            response.get(
                "summary",
                translate("ai.agents.summary_empty", language),
            )
        )
    )

    left_column, right_column = st.columns(2)

    with left_column:
        _render_text_items(
            translate("ai.agents.findings", language),
            _as_string_list(response.get("findings")),
            language,
        )

        _render_text_items(
            translate("ai.agents.risks", language),
            _as_string_list(response.get("risks")),
            language,
        )

    with right_column:
        _render_text_items(
            translate("ai.agents.actions", language),
            _as_string_list(response.get("recommendations")),
            language,
        )

        _render_text_items(
            translate("ai.agents.limitations_title", language),
            _as_string_list(response.get("limitations")),
            language,
        )

    st.markdown(f"#### {translate('ai.agents.evidence', language)}")

    _render_agent_evidence(_as_mapping_list(response.get("evidence")), language)

    st.warning(translate("ai.agents.warning", language))


def _render_frozen_agent_results(
    results: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render summaries from reproducible frozen workflow."""
    responses = _as_mapping(results.get("responses"))

    if not responses:
        st.info(translate("ai.agents.workflow_empty", language))
        return

    st.caption(
        translate(
            "ai.agents.workflow_metadata",
            language,
            model=results.get("model_name", "—"),
            retrieval=results.get("retrieval_strategy", "—"),
            basis=results.get("evaluation_basis", "—"),
        )
    )

    rows: list[dict[str, object]] = []

    for role in [
        "forecast",
        "risk",
        "allocation",
    ]:
        response = _as_mapping(responses.get(role))

        if not response:
            continue

        rows.append(
            {
                translate("ai.agents.column.role", language): _agent_type_label(
                    role, language
                ),
                translate("ai.agents.column.agent", language): (
                    _agent_name_label(
                        str(
                            response.get(
                                "agent_name",
                                "—",
                            )
                        ),
                        language,
                    )
                ),
                translate("ai.agents.column.summary", language): str(
                    response.get(
                        "summary",
                        "—",
                    )
                ),
            }
        )

    if rows:
        render_centered_dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )


def _run_agent(
    client: APIClient,
    *,
    agent_type: str,
    question: str,
    ticker: str,
    risk_preference: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Execute the selected decision-support agent."""
    if agent_type == "forecast":
        payload: dict[
            str,
            Any,
        ] = {
            "question": question,
            "response_language": language,
        }

        normalized_ticker = ticker.strip().upper()

        if normalized_ticker:
            payload["ticker"] = normalized_ticker

        return client.post(
            AGENT_FORECAST_PATH,
            payload,
            timeout_seconds=(GENERATIVE_TIMEOUT_SECONDS),
        )

    if agent_type == "risk":
        return client.post(
            AGENT_RISK_PATH,
            {
                "question": question,
                "response_language": language,
            },
            timeout_seconds=(GENERATIVE_TIMEOUT_SECONDS),
        )

    if agent_type == "allocation":
        return client.post(
            AGENT_ALLOCATION_PATH,
            {
                "question": question,
                "risk_preference": (risk_preference),
                "response_language": language,
            },
            timeout_seconds=(GENERATIVE_TIMEOUT_SECONDS),
        )

    raise ValueError("Unsupported agent type.")


def _agent_question_placeholder(
    agent_type: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Return context-aware analytical placeholder."""
    return translate(f"ai.agents.placeholder.{agent_type}", language)


def _render_agents_tab(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render interactive decision-support agents."""
    st.subheader(translate("ai.agents.heading", language))

    st.caption(translate("ai.agents.caption", language))

    try:
        status = client.get(AGENT_STATUS_PATH)
    except APIClientError as exc:
        st.error(translate("ai.agents.status_error", language))

        st.caption(str(exc))
        return

    _render_agent_status(status, language)

    if not bool(
        status.get(
            "runtime_inputs_available",
            False,
        )
    ):
        st.warning(translate("ai.agents.runtime_missing", language))

    st.divider()

    agent_types = [
        "forecast",
        "risk",
        "allocation",
    ]

    selected_agent = str(
        st.selectbox(
            translate("ai.agents.type", language),
            agent_types,
            format_func=lambda value: _agent_type_label(value, language),
            key="ai_insights_agent_type_v2",
            on_change=_clear_agent_response,
        )
    )

    question = st.text_area(
        translate("ai.agents.question", language),
        value="",
        placeholder=_agent_question_placeholder(selected_agent, language),
        height=105,
        key="ai_insights_agent_question_v2",
        on_change=_clear_agent_response,
    )

    ticker = ""

    if selected_agent == "forecast":
        ticker = st.text_input(
            "Ticker",
            value="",
            placeholder=translate("ai.agents.ticker_placeholder", language),
            help=translate("ai.agents.ticker_help", language),
            key="ai_insights_agent_ticker_v2",
            on_change=_clear_agent_response,
        )

    risk_preference = "balanced"

    if selected_agent == "allocation":
        preferences = [
            "defensive",
            "balanced",
            "growth",
        ]

        risk_preference = str(
            st.selectbox(
                translate("ai.agents.risk_preference", language),
                preferences,
                index=1,
                format_func=lambda value: _risk_preference_label(value, language),
                key=("ai_insights_risk_preference_v2"),
                on_change=(_clear_agent_response),
            )
        )

    submitted = st.button(
        translate("ai.agents.submit", language),
        type="primary",
        key="ai_insights_agent_submit_v2",
    )

    if submitted:
        normalized_question = question.strip()

        if not normalized_question:
            st.warning(translate("ai.agents.question_empty", language))
        else:
            try:
                with st.spinner(translate("ai.agents.spinner", language)):
                    response = _run_agent(
                        client,
                        agent_type=(selected_agent),
                        question=(normalized_question),
                        ticker=ticker,
                        risk_preference=(risk_preference),
                        language=language,
                    )
            except APITimeoutError:
                _render_timeout_message(
                    operation=translate("ai.agents.operation", language),
                    language=language,
                )
            except APIClientError as exc:
                st.error(translate("ai.agents.error", language))

                st.caption(str(exc))
            else:
                st.session_state[AGENT_RESPONSE_STATE_KEY] = response

    response_value = st.session_state.get(AGENT_RESPONSE_STATE_KEY)

    if isinstance(
        response_value,
        dict,
    ):
        response = cast(
            dict[str, Any],
            response_value,
        )

        response_question = str(
            response.get(
                "question",
                "",
            )
        ).strip()

        if question.strip() and response_question == question.strip():
            _render_agent_response(response, language)

    st.divider()

    try:
        frozen_results = client.get(AGENT_RESULTS_PATH)
    except APIClientError as exc:
        with st.expander(translate("ai.agents.workflow", language)):
            st.warning(translate("ai.agents.workflow_unavailable", language))

            st.caption(str(exc))
    else:
        with st.expander(translate("ai.agents.workflow", language)):
            _render_frozen_agent_results(frozen_results, language)

    with st.expander(translate("ai.agents.limitations", language)):
        st.markdown(translate("ai.agents.limitations_body", language))


def render_ai_insights_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render RAG and decision-support agent capabilities."""
    st.title(translate("ai.title", language))

    st.caption(translate("ai.caption", language))

    st.info(translate("ai.info", language))

    rag_tab, agents_tab = st.tabs(
        [
            translate("ai.tab.rag", language),
            translate("ai.tab.agents", language),
        ]
    )

    with rag_tab:
        _render_rag_tab(client, language)

    with agents_tab:
        _render_agents_tab(client, language)

"""Human-in-the-loop Decision Workspace for the Streamlit frontend."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any, Final, Literal, cast
from urllib.parse import urlencode

import pandas as pd
import streamlit as st

from frontend.api_client import APIClient, APIClientError
from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate
from frontend.presentation import (
    DEFAULT_PRESENTATION_TIMEZONE,
    SUPPORTED_PRESENTATION_TIMEZONES,
    render_centered_dataframe,
)

DECISION_PROFILES_PATH: Final = "/api/v1/decision/profiles"
DECISION_RECOMMENDATION_PATH: Final = "/api/v1/decision/recommendations/{profile_type}"
DECISION_EVALUATION_PATH: Final = "/api/v1/decision/evaluate"
DECISION_NARRATIVE_PATH: Final = "/api/v1/agents/decision"
DECISION_REPORT_PATH: Final = "/api/v1/reports/decision"
DECISION_RESULT_STATE_KEY: Final = "decision_workspace_result"
DECISION_PREVIOUS_RESULT_STATE_KEY: Final = "decision_workspace_previous_result"
DECISION_BASELINE_RESULT_STATE_KEY: Final = "decision_workspace_baseline_result"
DECISION_CUSTOM_APPLIED_STATE_KEY: Final = "decision_workspace_custom_applied"
DECISION_EVALUATION_REQUEST_STATE_KEY: Final = "decision_workspace_request"
DECISION_NARRATIVE_STATE_KEY: Final = "decision_workspace_narrative"
DECISION_REPORT_STATE_KEY: Final = "decision_workspace_report"

COMPONENT_NAMES: Final = ("return", "volatility", "drawdown", "diversification")
LiquidityNeed = Literal["low", "medium", "high"]
LIQUIDITY_NAMES: Final[tuple[LiquidityNeed, ...]] = ("low", "medium", "high")
CONSTRAINT_NAMES: Final = (
    "maximum_annualized_volatility",
    "maximum_drawdown",
    "maximum_concentration_index",
    "minimum_effective_assets",
    "minimum_diversification_ratio",
    "minimum_cash_reserve",
)


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return cast(dict[str, Any], value)


def _as_mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_liquidity_need(value: Any) -> LiquidityNeed:
    if value in LIQUIDITY_NAMES:
        return cast(LiquidityNeed, value)
    return "medium"


def _scenario_label(
    name: str | None,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    if not name:
        return translate("decision.scenario.none", language)
    key = f"decision.scenario.{name}"
    try:
        return translate(key, language)
    except KeyError:
        return name.replace("_", " ").title()


def _profile_label(
    name: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    try:
        return translate(f"decision.profile.{name}", language)
    except KeyError:
        return name.title()


def _component_label(name: str, language: LanguageCode) -> str:
    return translate(f"decision.component.{name}", language)


def _liquidity_label(name: str, language: LanguageCode) -> str:
    return translate(f"decision.liquidity.{name}", language)


def _recommendation_path(
    profile_type: str,
    available_liquidity_fraction: float,
) -> str:
    query = urlencode(
        {"available_liquidity_fraction": f"{available_liquidity_fraction:.4f}"}
    )
    return DECISION_RECOMMENDATION_PATH.format(profile_type=profile_type) + f"?{query}"


def _load_profiles(client: APIClient) -> list[dict[str, Any]]:
    payload = client.get(DECISION_PROFILES_PATH)
    return _as_mapping_list(payload.get("profiles"))


def _load_recommendation(
    client: APIClient,
    profile_type: str,
    available_liquidity_fraction: float,
) -> dict[str, Any]:
    return client.get(_recommendation_path(profile_type, available_liquidity_fraction))


def _load_custom_recommendation(
    client: APIClient,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return client.post(DECISION_EVALUATION_PATH, payload)


def _load_decision_narrative(
    client: APIClient,
    *,
    question: str,
    evaluation: dict[str, Any],
    response_language: LanguageCode = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Request an optional narrative for a server-recomputed result."""
    return client.post(
        DECISION_NARRATIVE_PATH,
        {
            "question": question,
            "evaluation": evaluation,
            "response_language": response_language,
        },
        timeout_seconds=120.0,
    )


def _load_decision_report(
    client: APIClient,
    *,
    question: str,
    evaluation: dict[str, Any],
    presentation_timezone: str,
    response_language: LanguageCode = DEFAULT_LANGUAGE,
) -> dict[str, Any]:
    """Generate exact report downloads from a server-recomputed decision."""
    return client.post(
        DECISION_REPORT_PATH,
        {
            "question": question,
            "evaluation": evaluation,
            "presentation_timezone": presentation_timezone,
            "response_language": response_language,
        },
        timeout_seconds=120.0,
    )


def _validated_report_files(
    payload: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], bytes]]:
    """Decode downloads only after checking size, format and SHA-256."""
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("Odpowiedź raportu nie zawiera listy plików.")
    result: dict[str, tuple[dict[str, Any], bytes]] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("Metadane pliku raportu są niepoprawne.")
        format_name = item.get("format")
        encoded = item.get("content_base64")
        if format_name not in {"json", "markdown", "pdf"} or not isinstance(
            encoded, str
        ):
            raise ValueError("Format pliku raportu jest niepoprawny.")
        if format_name in result:
            raise ValueError("Odpowiedź raportu zawiera powtórzony format.")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError(
                "Treść pliku raportu nie jest poprawnym Base64."
            ) from error
        digest = hashlib.sha256(content).hexdigest()
        if item.get("bytes") != len(content) or item.get("sha256") != digest:
            raise ValueError("Kontrola integralności pliku raportu nie powiodła się.")
        result[str(format_name)] = (cast(dict[str, Any], item), content)
    if set(result) != {"json", "markdown", "pdf"}:
        raise ValueError("Odpowiedź nie zawiera wszystkich formatów raportu.")
    return result


def _build_preset_evaluation_payload(
    profile_type: str,
    available_liquidity_fraction: float,
) -> dict[str, Any]:
    return {
        "profile": {"profile_type": profile_type},
        "available_liquidity_fraction": available_liquidity_fraction,
    }


def _build_custom_evaluation_payload(
    *,
    priorities: dict[str, float],
    constraints: dict[str, float],
    liquidity_need: str,
    investment_horizon_years: int,
    available_liquidity_fraction: float,
) -> dict[str, Any]:
    """Build the complete API payload for explicit custom assumptions."""
    return {
        "profile": {
            "profile_type": "custom",
            "priorities": {
                "return_priority": priorities["return"],
                "volatility_priority": priorities["volatility"],
                "drawdown_priority": priorities["drawdown"],
                "diversification_priority": priorities["diversification"],
            },
            "constraints": dict(constraints),
            "liquidity_need": liquidity_need,
            "investment_horizon_years": investment_horizon_years,
        },
        "available_liquidity_fraction": available_liquidity_fraction,
    }


def _build_ranking_frame(
    result: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in _as_mapping_list(result.get("scenarios")):
        violations = []
        for item in _as_string_list(scenario.get("constraint_violations")):
            try:
                violations.append(translate(f"decision.violation.{item}", language))
            except KeyError:
                violations.append(item)
        total_score = scenario.get("total_score")
        rows.append(
            {
                translate("decision.column.rank", language): scenario.get("rank")
                or "—",
                translate("decision.column.scenario", language): _scenario_label(
                    str(scenario.get("name", "")), language
                ),
                translate("decision.column.admissible", language): translate(
                    "decision.yes" if scenario.get("admissible") else "decision.no",
                    language,
                ),
                translate("decision.column.score", language): (
                    f"{_as_float(total_score):.3f}" if total_score is not None else "—"
                ),
                translate("decision.column.violations", language): ", ".join(violations)
                if violations
                else "—",
            }
        )
    return pd.DataFrame(rows)


def _build_components_frame(
    result: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    recommended = result.get("recommended_scenario")
    scenarios = _as_mapping_list(result.get("scenarios"))
    selected = next(
        (scenario for scenario in scenarios if scenario.get("name") == recommended),
        None,
    )
    if selected is None:
        return pd.DataFrame()
    normalized = _as_mapping(selected.get("normalized_components"))
    weighted = _as_mapping(selected.get("weighted_contributions"))
    return pd.DataFrame(
        [
            {
                translate("decision.column.criterion", language): _component_label(
                    name, language
                ),
                translate("decision.column.normalized", language): _as_float(value),
                translate("decision.column.contribution", language): _as_float(
                    weighted.get(name)
                ),
            }
            for name, value in normalized.items()
        ]
    )


def _recommendation_changed(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> bool:
    return bool(
        previous
        and previous.get("recommended_scenario") != current.get("recommended_scenario")
    )


def _build_comparison_frame(
    baseline: dict[str, Any],
    current: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> pd.DataFrame:
    """Compare every scenario before and after changing assumptions."""
    baseline_scenarios = {
        str(item.get("name")): item
        for item in _as_mapping_list(baseline.get("scenarios"))
    }
    current_scenarios = {
        str(item.get("name")): item
        for item in _as_mapping_list(current.get("scenarios"))
    }
    names = sorted(set(baseline_scenarios) | set(current_scenarios))

    def value(
        scenario: dict[str, Any] | None,
        key: str,
    ) -> object:
        if scenario is None:
            return "—"
        item = scenario.get(key)
        if key == "admissible":
            return translate("decision.yes" if item else "decision.no", language)
        if key == "total_score":
            return f"{_as_float(item):.3f}" if item is not None else "—"
        return item if item is not None else "—"

    return pd.DataFrame(
        [
            {
                translate("decision.column.scenario", language): _scenario_label(
                    name, language
                ),
                translate("decision.column.before_admissible", language): value(
                    baseline_scenarios.get(name), "admissible"
                ),
                translate("decision.column.before_rank", language): value(
                    baseline_scenarios.get(name), "rank"
                ),
                translate("decision.column.before_score", language): value(
                    baseline_scenarios.get(name), "total_score"
                ),
                translate("decision.column.after_admissible", language): value(
                    current_scenarios.get(name), "admissible"
                ),
                translate("decision.column.after_rank", language): value(
                    current_scenarios.get(name), "rank"
                ),
                translate("decision.column.after_score", language): value(
                    current_scenarios.get(name), "total_score"
                ),
            }
            for name in names
        ]
    )


def _build_change_explanations(
    baseline: dict[str, Any],
    current: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> list[str]:
    """Explain observable deterministic differences without causal overclaiming."""
    explanations: list[str] = []
    baseline_recommendation = baseline.get("recommended_scenario")
    current_recommendation = current.get("recommended_scenario")
    if baseline_recommendation == current_recommendation:
        explanations.append(
            translate(
                "decision.change.unchanged",
                language,
                name=_scenario_label(
                    cast(str | None, current_recommendation), language
                ),
            )
        )
    else:
        explanations.append(
            translate(
                "decision.change.changed",
                language,
                before=_scenario_label(
                    cast(str | None, baseline_recommendation), language
                ),
                after=_scenario_label(
                    cast(str | None, current_recommendation), language
                ),
            )
        )

    baseline_profile = _as_mapping(baseline.get("profile"))
    current_profile = _as_mapping(current.get("profile"))
    baseline_priorities = _as_mapping(baseline_profile.get("priorities"))
    current_priorities = _as_mapping(current_profile.get("priorities"))
    for name in COMPONENT_NAMES:
        before = _as_float(baseline_priorities.get(name))
        after = _as_float(current_priorities.get(name))
        if abs(before - after) > 1e-9:
            explanations.append(
                f"Priorytet „{_component_label(name, language)}” zmieniono "
                f"z {before:.0%} na {after:.0%}."
                if language == "pl"
                else f"Priority “{_component_label(name, language)}” changed "
                f"from {before:.0%} to {after:.0%}."
            )

    baseline_constraints = _as_mapping(baseline_profile.get("constraints"))
    current_constraints = _as_mapping(current_profile.get("constraints"))
    percentage_constraints = {
        "maximum_annualized_volatility",
        "maximum_drawdown",
        "maximum_concentration_index",
        "minimum_cash_reserve",
    }
    for name in CONSTRAINT_NAMES:
        label = translate(f"decision.constraint.{name}", language)
        before = _as_float(baseline_constraints.get(name))
        after = _as_float(current_constraints.get(name))
        if abs(before - after) <= 1e-9:
            continue
        if name in percentage_constraints:
            formatted_before = f"{before:.0%}"
            formatted_after = f"{after:.0%}"
        else:
            formatted_before = f"{before:g}"
            formatted_after = f"{after:g}"
        explanations.append(
            f"Parametr „{label}” zmieniono z {formatted_before} na {formatted_after}."
            if language == "pl"
            else f"Parameter “{label}” changed from {formatted_before} "
            f"to {formatted_after}."
        )

    baseline_scenarios = {
        str(item.get("name")): bool(item.get("admissible"))
        for item in _as_mapping_list(baseline.get("scenarios"))
    }
    current_scenarios = {
        str(item.get("name")): bool(item.get("admissible"))
        for item in _as_mapping_list(current.get("scenarios"))
    }
    for name in sorted(set(baseline_scenarios) & set(current_scenarios)):
        if baseline_scenarios[name] and not current_scenarios[name]:
            explanations.append(
                translate(
                    "decision.change.rejected",
                    language,
                    name=_scenario_label(name, language),
                )
            )
        elif not baseline_scenarios[name] and current_scenarios[name]:
            explanations.append(
                translate(
                    "decision.change.admitted",
                    language,
                    name=_scenario_label(name, language),
                )
            )
    return explanations


def _profile_form_defaults(
    profile: dict[str, Any],
) -> tuple[
    dict[str, int],
    dict[str, float],
    LiquidityNeed,
    int,
]:
    priorities = _as_mapping(profile.get("priorities"))
    constraints = _as_mapping(profile.get("constraints"))
    return (
        {
            name: int(round(100.0 * _as_float(priorities.get(name))))
            for name in COMPONENT_NAMES
        },
        {
            "maximum_annualized_volatility": 100.0
            * _as_float(constraints.get("maximum_annualized_volatility")),
            "maximum_drawdown": 100.0 * _as_float(constraints.get("maximum_drawdown")),
            "maximum_concentration_index": 100.0
            * _as_float(constraints.get("maximum_concentration_index")),
            "minimum_effective_assets": _as_float(
                constraints.get("minimum_effective_assets"), 1.0
            ),
            "minimum_diversification_ratio": _as_float(
                constraints.get("minimum_diversification_ratio"), 1.0
            ),
            "minimum_cash_reserve": 100.0
            * _as_float(constraints.get("minimum_cash_reserve")),
        },
        _as_liquidity_need(profile.get("liquidity_need", "medium")),
        int(profile.get("investment_horizon_years", 5)),
    )


def _render_profile_summary(
    profile: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    priorities = _as_mapping(profile.get("priorities"))
    constraints = _as_mapping(profile.get("constraints"))
    st.markdown(f"#### {translate('decision.profile.active', language)}")
    priority_columns = st.columns(4)
    for column, key in zip(
        priority_columns,
        ("return", "volatility", "drawdown", "diversification"),
        strict=True,
    ):
        with column:
            st.metric(
                _component_label(key, language),
                f"{_as_float(priorities.get(key)):.0%}",
            )
    with st.expander(translate("decision.profile.constraints", language)):
        st.json(constraints)


def _render_custom_assumptions_form(
    profile: dict[str, Any],
    *,
    available_liquidity_fraction: float,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> tuple[bool, dict[str, Any] | None]:
    priority_defaults, constraint_defaults, liquidity_default, horizon_default = (
        _profile_form_defaults(profile)
    )
    profile_type = str(profile.get("profile_type", "preset"))
    st.subheader(translate("decision.custom.title", language))
    st.caption(translate("decision.custom.caption", language))
    with st.form(f"decision_custom_assumptions_{profile_type}"):
        st.markdown(f"##### {translate('decision.custom.priorities', language)}")
        priority_columns = st.columns(4)
        priority_percentages: dict[str, int] = {}
        for column, name in zip(
            priority_columns,
            COMPONENT_NAMES,
            strict=True,
        ):
            with column:
                priority_percentages[name] = st.number_input(
                    _component_label(name, language),
                    min_value=0,
                    max_value=100,
                    value=priority_defaults[name],
                    step=5,
                    key=f"decision_custom_priority_{profile_type}_{name}",
                )
        priority_sum = sum(priority_percentages.values())
        st.caption(
            translate("decision.custom.priority_sum", language, value=priority_sum)
        )

        st.markdown(f"##### {translate('decision.custom.constraints', language)}")
        first_row = st.columns(3)
        with first_row[0]:
            maximum_volatility = st.number_input(
                translate("decision.custom.max_volatility", language),
                min_value=1.0,
                max_value=100.0,
                value=constraint_defaults["maximum_annualized_volatility"],
                step=1.0,
                key=f"decision_custom_volatility_{profile_type}",
            )
        with first_row[1]:
            maximum_drawdown = st.number_input(
                translate("decision.custom.max_drawdown", language),
                min_value=0.0,
                max_value=100.0,
                value=constraint_defaults["maximum_drawdown"],
                step=1.0,
                key=f"decision_custom_drawdown_{profile_type}",
            )
        with first_row[2]:
            maximum_concentration = st.number_input(
                translate("decision.custom.max_concentration", language),
                min_value=1.0,
                max_value=100.0,
                value=constraint_defaults["maximum_concentration_index"],
                step=1.0,
                key=f"decision_custom_concentration_{profile_type}",
            )

        second_row = st.columns(3)
        with second_row[0]:
            minimum_effective_assets = st.number_input(
                translate("decision.custom.min_assets", language),
                min_value=1.0,
                max_value=50.0,
                value=constraint_defaults["minimum_effective_assets"],
                step=0.5,
                key=f"decision_custom_assets_{profile_type}",
            )
        with second_row[1]:
            minimum_diversification = st.number_input(
                translate("decision.custom.min_diversification", language),
                min_value=0.1,
                max_value=10.0,
                value=constraint_defaults["minimum_diversification_ratio"],
                step=0.1,
                key=f"decision_custom_diversification_{profile_type}",
            )
        with second_row[2]:
            minimum_cash_reserve = st.number_input(
                translate("decision.custom.min_cash", language),
                min_value=0.0,
                max_value=100.0,
                value=constraint_defaults["minimum_cash_reserve"],
                step=1.0,
                key=f"decision_custom_reserve_{profile_type}",
            )

        context_columns = st.columns(2)
        liquidity_options = list(LIQUIDITY_NAMES)
        with context_columns[0]:
            liquidity_need = st.selectbox(
                translate("decision.custom.liquidity_need", language),
                liquidity_options,
                index=liquidity_options.index(liquidity_default),
                format_func=lambda value: _liquidity_label(value, language),
                key=f"decision_custom_liquidity_need_{profile_type}",
            )
        with context_columns[1]:
            investment_horizon = st.number_input(
                translate("decision.custom.horizon", language),
                min_value=1,
                max_value=50,
                value=horizon_default,
                step=1,
                key=f"decision_custom_horizon_{profile_type}",
            )
        submitted = st.form_submit_button(
            translate("decision.custom.submit", language),
            type="primary",
            key=f"decision_custom_submit_{profile_type}",
        )

    if not submitted:
        return False, None
    if priority_sum != 100:
        st.error(translate("decision.custom.sum_error", language))
        return True, None
    return (
        True,
        _build_custom_evaluation_payload(
            priorities={
                name: percentage / 100.0
                for name, percentage in priority_percentages.items()
            },
            constraints={
                "maximum_annualized_volatility": maximum_volatility / 100.0,
                "maximum_drawdown": maximum_drawdown / 100.0,
                "maximum_concentration_index": maximum_concentration / 100.0,
                "minimum_effective_assets": minimum_effective_assets,
                "minimum_diversification_ratio": minimum_diversification,
                "minimum_cash_reserve": minimum_cash_reserve / 100.0,
            },
            liquidity_need=str(liquidity_need),
            investment_horizon_years=int(investment_horizon),
            available_liquidity_fraction=available_liquidity_fraction,
        ),
    )


def _render_before_after_comparison(
    baseline: dict[str, Any],
    current: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    st.subheader(translate("decision.comparison.title", language))
    before_column, after_column = st.columns(2)
    with before_column:
        st.metric(
            translate("decision.comparison.before", language),
            _scenario_label(
                cast(str | None, baseline.get("recommended_scenario")), language
            ),
        )
    with after_column:
        st.metric(
            translate("decision.comparison.after", language),
            _scenario_label(
                cast(str | None, current.get("recommended_scenario")), language
            ),
        )
    for explanation in _build_change_explanations(baseline, current, language):
        st.write(f"- {explanation}")
    render_centered_dataframe(
        _build_comparison_frame(baseline, current, language),
        width="stretch",
        hide_index=True,
    )
    st.caption(translate("decision.comparison.caption", language))


def _render_result(
    result: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    scenarios = _as_mapping_list(result.get("scenarios"))
    admissible_count = sum(bool(item.get("admissible")) for item in scenarios)
    recommended = result.get("recommended_scenario")
    provenance = _as_mapping(result.get("provenance"))

    recommendation_column, admissible_column, rejected_column, source_column = (
        st.columns(4)
    )
    with recommendation_column:
        st.metric(
            translate("decision.metric.recommended", language),
            _scenario_label(cast(str | None, recommended), language),
        )
    with admissible_column:
        st.metric(
            translate("decision.metric.admissible", language), str(admissible_count)
        )
    with rejected_column:
        st.metric(
            translate("decision.metric.rejected", language),
            str(len(scenarios) - admissible_count),
        )
    with source_column:
        st.metric(
            translate("decision.metric.source", language),
            str(provenance.get("evaluation_split", "—")),
        )

    if recommended is None:
        st.warning(translate("decision.no_admissible", language))
    else:
        st.success(
            translate(
                "decision.success",
                language,
                name=_scenario_label(str(recommended), language),
            )
        )

    st.subheader(translate("decision.ranking", language))
    render_centered_dataframe(
        _build_ranking_frame(result, language), width="stretch", hide_index=True
    )

    components = _build_components_frame(result, language)
    if not components.empty:
        st.subheader(translate("decision.components", language))
        render_centered_dataframe(components, width="stretch", hide_index=True)

    with st.expander(translate("decision.governance", language)):
        st.json(provenance)
        st.write(
            translate("decision.governance.deterministic", language),
            bool(result.get("deterministic")),
        )
        st.write(
            translate("decision.governance.llm", language),
            bool(result.get("llm_used_for_ranking")),
        )
        st.write(
            translate("decision.governance.weights", language),
            bool(result.get("portfolio_weights_modified")),
        )


def _render_agent_narrative(
    narrative: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    st.subheader(translate("decision.narrative.result", language))
    st.write(str(narrative.get("summary", "")))

    sections = (
        (translate("decision.narrative.findings", language), "findings"),
        (translate("decision.narrative.risks", language), "risks"),
        (translate("decision.narrative.steps", language), "recommendations"),
        (translate("decision.narrative.limitations", language), "limitations"),
    )
    for title, key in sections:
        values = _as_string_list(narrative.get(key))
        if values:
            st.markdown(f"##### {title}")
            for value in values:
                st.write(f"- {value}")

    evidence = _as_mapping_list(narrative.get("evidence"))
    with st.expander(translate("decision.narrative.sources", language)):
        for item in evidence:
            st.write(f"- {item.get('identifier', '—')}: {item.get('description', '—')}")


def _render_narrative_workspace(
    client: APIClient,
    evaluation_request: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    st.subheader(translate("decision.narrative.title", language))
    st.caption(translate("decision.narrative.caption", language))
    with st.form("decision_narrative_form"):
        question = st.text_area(
            translate("decision.narrative.question", language),
            value=translate("decision.narrative.default_question", language),
            max_chars=2000,
            key="decision_narrative_question",
        )
        submitted = st.form_submit_button(
            translate("decision.narrative.submit", language)
        )

    if submitted:
        try:
            payload = _load_decision_narrative(
                client,
                question=question,
                evaluation=evaluation_request,
                response_language=language,
            )
        except APIClientError as exc:
            st.error(translate("decision.narrative.error", language))
            st.caption(str(exc))
        else:
            st.session_state[DECISION_NARRATIVE_STATE_KEY] = payload.get("narrative")

    narrative = st.session_state.get(DECISION_NARRATIVE_STATE_KEY)
    if isinstance(narrative, dict):
        _render_agent_narrative(cast(dict[str, Any], narrative), language)


def _render_decision_report_workspace(
    client: APIClient,
    evaluation_request: dict[str, Any],
    language: LanguageCode = DEFAULT_LANGUAGE,
    presentation_timezone: str = DEFAULT_PRESENTATION_TIMEZONE,
) -> None:
    """Generate and expose integrity-checked report downloads."""
    st.subheader(translate("decision.report.title", language))
    st.caption(translate("decision.report.caption", language))
    narrative = st.session_state.get(DECISION_NARRATIVE_STATE_KEY)
    default_question = (
        str(narrative.get("question"))
        if isinstance(narrative, dict) and narrative.get("question")
        else (translate("decision.narrative.default_question", language))
    )
    timezones = SUPPORTED_PRESENTATION_TIMEZONES
    with st.form("decision_report_form"):
        question = st.text_area(
            translate("decision.report.scope", language),
            value=default_question,
            max_chars=2000,
            key="decision_report_question",
        )
        presentation_timezone = st.selectbox(
            translate("decision.report.timezone", language),
            timezones,
            index=timezones.index(presentation_timezone),
            key="decision_report_timezone",
        )
        submitted = st.form_submit_button(
            translate("decision.report.submit", language),
            type="primary",
        )

    if submitted:
        try:
            payload = _load_decision_report(
                client,
                question=question,
                evaluation=evaluation_request,
                presentation_timezone=str(presentation_timezone),
                response_language=language,
            )
            _validated_report_files(payload)
        except (APIClientError, ValueError) as exc:
            st.error(translate("decision.report.error", language))
            st.caption(str(exc))
        else:
            st.session_state[DECISION_REPORT_STATE_KEY] = payload

    payload_value = st.session_state.get(DECISION_REPORT_STATE_KEY)
    if not isinstance(payload_value, dict):
        return
    payload = cast(dict[str, Any], payload_value)
    try:
        files = _validated_report_files(payload)
    except ValueError as exc:
        st.error(translate("decision.report.integrity_error", language))
        st.caption(str(exc))
        return

    report = _as_mapping(payload.get("report"))
    metadata = _as_mapping(report.get("metadata"))
    st.success(translate("decision.report.ready", language))
    st.caption(
        translate(
            "decision.report.metadata",
            language,
            source=metadata.get("evaluation_split", "—"),
            test=metadata.get("test_split_used", "—"),
            time=metadata.get("generated_at_utc", "—"),
        )
    )
    columns = st.columns(4)
    labels = {
        "json": translate("decision.report.download_json", language),
        "markdown": translate("decision.report.download_markdown", language),
        "pdf": translate("decision.report.download_pdf", language),
    }
    for column, format_name in zip(columns[:3], labels, strict=True):
        item, content = files[format_name]
        with column:
            st.download_button(
                labels[format_name],
                data=content,
                file_name=str(item.get("file_name")),
                mime=str(item.get("media_type")),
                key=f"decision_report_download_{format_name}",
            )
    manifest = _as_mapping(payload.get("manifest"))
    with columns[3]:
        st.download_button(
            translate("decision.report.download_manifest", language),
            data=json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            file_name="decision_report_manifest.json",
            mime="application/json",
            key="decision_report_download_manifest",
        )


def render_decision_workspace_page(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
    presentation_timezone: str = DEFAULT_PRESENTATION_TIMEZONE,
) -> None:
    """Render the first human-in-the-loop decision workspace."""
    st.title(translate("decision.title", language))
    st.caption(translate("decision.caption", language))
    st.info(translate("decision.local_info", language))

    try:
        profiles = _load_profiles(client)
    except APIClientError as exc:
        st.error(translate("decision.error.profiles", language))
        st.caption(str(exc))
        return
    if not profiles:
        st.warning(translate("decision.error.no_profiles", language))
        return

    profile_by_type = {
        str(profile.get("profile_type")): profile for profile in profiles
    }
    profile_types = list(profile_by_type)
    default_index = (
        profile_types.index("balanced") if "balanced" in profile_types else 0
    )

    with st.form("decision_workspace_form"):
        selected_profile = st.selectbox(
            translate("decision.profile.selector", language),
            profile_types,
            index=default_index,
            format_func=lambda item: _profile_label(item, language),
            key="decision_workspace_profile",
        )
        available_liquidity_percent = st.slider(
            translate("decision.liquidity.outside", language),
            min_value=0,
            max_value=50,
            value=10,
            step=1,
            format="%d%%",
            key="decision_workspace_liquidity",
            help=translate("decision.liquidity.help", language),
        )
        submitted = st.form_submit_button(
            translate("decision.calculate", language), type="primary"
        )

    selected_profile_data = profile_by_type[str(selected_profile)]
    _render_profile_summary(selected_profile_data, language)

    if submitted or DECISION_RESULT_STATE_KEY not in st.session_state:
        preset_evaluation_request = _build_preset_evaluation_payload(
            str(selected_profile),
            float(available_liquidity_percent) / 100.0,
        )
        try:
            result = _load_recommendation(
                client,
                str(selected_profile),
                float(available_liquidity_percent) / 100.0,
            )
        except APIClientError as exc:
            st.error(translate("decision.error.ranking", language))
            st.caption(str(exc))
            return
        previous = st.session_state.get(DECISION_RESULT_STATE_KEY)
        if isinstance(previous, dict):
            st.session_state[DECISION_PREVIOUS_RESULT_STATE_KEY] = previous
        st.session_state[DECISION_RESULT_STATE_KEY] = result
        st.session_state[DECISION_BASELINE_RESULT_STATE_KEY] = result
        st.session_state[DECISION_CUSTOM_APPLIED_STATE_KEY] = False
        st.session_state[DECISION_EVALUATION_REQUEST_STATE_KEY] = (
            preset_evaluation_request
        )
        st.session_state.pop(DECISION_NARRATIVE_STATE_KEY, None)
        st.session_state.pop(DECISION_REPORT_STATE_KEY, None)

    result_value = st.session_state.get(DECISION_RESULT_STATE_KEY)
    if not isinstance(result_value, dict):
        st.warning(translate("decision.error.no_result", language))
        return
    result = cast(dict[str, Any], result_value)
    previous_value = st.session_state.get(DECISION_PREVIOUS_RESULT_STATE_KEY)
    previous = (
        cast(dict[str, Any], previous_value)
        if isinstance(previous_value, dict)
        else None
    )
    if previous is not None and _recommendation_changed(previous, result):
        st.warning(
            translate(
                "decision.change.warning",
                language,
                before=_scenario_label(
                    cast(str | None, previous.get("recommended_scenario")), language
                ),
                after=_scenario_label(
                    cast(str | None, result.get("recommended_scenario")), language
                ),
            )
        )
    baseline_value = st.session_state.get(DECISION_BASELINE_RESULT_STATE_KEY)
    custom_applied = bool(
        st.session_state.get(DECISION_CUSTOM_APPLIED_STATE_KEY, False)
    )
    baseline = (
        cast(dict[str, Any], baseline_value)
        if isinstance(baseline_value, dict)
        else result
    )
    _render_result(baseline, language)

    custom_submitted, custom_payload = _render_custom_assumptions_form(
        selected_profile_data,
        available_liquidity_fraction=float(available_liquidity_percent) / 100.0,
        language=language,
    )
    if custom_submitted and custom_payload is not None:
        try:
            custom_result = _load_custom_recommendation(client, custom_payload)
        except APIClientError as exc:
            st.error(translate("decision.error.custom", language))
            st.caption(str(exc))
        else:
            st.session_state[DECISION_RESULT_STATE_KEY] = custom_result
            st.session_state[DECISION_CUSTOM_APPLIED_STATE_KEY] = True
            st.session_state[DECISION_EVALUATION_REQUEST_STATE_KEY] = custom_payload
            st.session_state.pop(DECISION_NARRATIVE_STATE_KEY, None)
            st.session_state.pop(DECISION_REPORT_STATE_KEY, None)
            result = custom_result

    custom_applied = bool(
        st.session_state.get(DECISION_CUSTOM_APPLIED_STATE_KEY, False)
    )
    if custom_applied:
        _render_before_after_comparison(baseline, result, language)
        st.subheader(translate("decision.custom.result", language))
        _render_result(result, language)

    evaluation_request = st.session_state.get(DECISION_EVALUATION_REQUEST_STATE_KEY)
    if isinstance(evaluation_request, dict):
        _render_narrative_workspace(
            client,
            cast(dict[str, Any], evaluation_request),
            language,
        )
        _render_decision_report_workspace(
            client,
            cast(dict[str, Any], evaluation_request),
            language,
            presentation_timezone,
        )

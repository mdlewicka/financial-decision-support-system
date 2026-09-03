"""Streamlit frontend for the Financial Decision Support System."""

import os
from typing import Any, Final, cast

import streamlit as st

from frontend.about_view import render_about_page
from frontend.accessibility import render_accessibility_styles
from frontend.ai_insights_view import (
    render_ai_insights_page,
)
from frontend.api_client import (
    DEFAULT_API_URL,
    APIClient,
    APIClientError,
)
from frontend.branding import render_branding_styles
from frontend.dashboard_view import (
    render_dashboard_page,
)
from frontend.decision_workspace_view import (
    render_decision_workspace_page,
)
from frontend.forecast_view import (
    render_forecast_page,
)
from frontend.governance_view import (
    render_governance_page,
)
from frontend.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_NAMES,
    SUPPORTED_LANGUAGES,
    LanguageCode,
    translate,
)
from frontend.market_intelligence_view import (
    render_market_intelligence_page,
)
from frontend.optimization_view import (
    render_optimization_page,
)
from frontend.presentation import (
    DEFAULT_PRESENTATION_TIMEZONE,
    SUPPORTED_PRESENTATION_TIMEZONES,
)
from frontend.reports_view import (
    render_reports_page,
)
from frontend.risk_view import (
    render_risk_page,
)

APPLICATION_TITLE: Final = "System wspomagania decyzji finansowych"

APPLICATION_SUBTITLE: Final = "Financial Decision Support System"

API_URL_ENVIRONMENT_VARIABLE: Final = "FINANCIAL_DECISION_SUPPORT_API_URL"

SELECTED_PAGE_STATE_KEY: Final = "selected_application_page"
LANGUAGE_STATE_KEY: Final = "presentation_language"
TIMEZONE_STATE_KEY: Final = "presentation_timezone"

ABOUT_PAGE: Final = "About"
DASHBOARD_PAGE: Final = "Dashboard"
FORECASTING_PAGE: Final = "Forecasting"
RISK_SCENARIOS_PAGE: Final = "Risk and Scenarios"
OPTIMIZATION_PAGE: Final = "Optimization"
DECISION_WORKSPACE_PAGE: Final = "Decision Workspace"
MARKET_INTELLIGENCE_PAGE: Final = "Market Intelligence"
AI_INSIGHTS_PAGE: Final = "AI Insights"
GOVERNANCE_PAGE: Final = "Explainability and Governance"
REPORTS_PAGE: Final = "Reports"

NAVIGATION_OPTIONS: Final[tuple[str, ...]] = (
    ABOUT_PAGE,
    DASHBOARD_PAGE,
    FORECASTING_PAGE,
    RISK_SCENARIOS_PAGE,
    OPTIMIZATION_PAGE,
    DECISION_WORKSPACE_PAGE,
    MARKET_INTELLIGENCE_PAGE,
    AI_INSIGHTS_PAGE,
    GOVERNANCE_PAGE,
    REPORTS_PAGE,
)

NAVIGATION_ICONS: Final[dict[str, str]] = {
    ABOUT_PAGE: (":material/info:"),
    DASHBOARD_PAGE: (":material/dashboard:"),
    FORECASTING_PAGE: (":material/show_chart:"),
    RISK_SCENARIOS_PAGE: (":material/security:"),
    OPTIMIZATION_PAGE: (":material/tune:"),
    DECISION_WORKSPACE_PAGE: (":material/tune:"),
    MARKET_INTELLIGENCE_PAGE: (":material/query_stats:"),
    AI_INSIGHTS_PAGE: (":material/psychology:"),
    GOVERNANCE_PAGE: (":material/policy:"),
    REPORTS_PAGE: (":material/description:"),
}

NAVIGATION_TRANSLATION_KEYS: Final[dict[str, str]] = {
    ABOUT_PAGE: "navigation.about",
    DASHBOARD_PAGE: "navigation.dashboard",
    FORECASTING_PAGE: "navigation.forecasting",
    RISK_SCENARIOS_PAGE: "navigation.risk",
    OPTIMIZATION_PAGE: "navigation.optimization",
    DECISION_WORKSPACE_PAGE: "navigation.decision",
    MARKET_INTELLIGENCE_PAGE: "navigation.market",
    AI_INSIGHTS_PAGE: "navigation.ai",
    GOVERNANCE_PAGE: "navigation.governance",
    REPORTS_PAGE: "navigation.reports",
}


def configure_page() -> None:
    """Configure global Streamlit page metadata."""
    st.set_page_config(
        page_title=APPLICATION_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def create_api_client() -> APIClient:
    """Create the frontend HTTP client."""
    api_url = os.getenv(
        API_URL_ENVIRONMENT_VARIABLE,
        DEFAULT_API_URL,
    )

    return APIClient(
        base_url=api_url,
    )


def load_api_status(
    client: APIClient,
) -> tuple[
    bool,
    dict[str, Any] | None,
    dict[str, Any] | None,
    str | None,
]:
    """Load backend health and API information."""
    try:
        health = client.get_system_health()

        information = client.get_system_info()
    except APIClientError as exc:
        return (
            False,
            None,
            None,
            str(exc),
        )

    return (
        True,
        health,
        information,
        None,
    )


def _initialize_selected_page() -> None:
    """Initialize navigation state."""
    if SELECTED_PAGE_STATE_KEY not in st.session_state:
        st.session_state[SELECTED_PAGE_STATE_KEY] = ABOUT_PAGE


def _initialize_presentation_preferences() -> None:
    """Initialize stable language and time-zone session values."""
    if LANGUAGE_STATE_KEY not in st.session_state:
        st.session_state[LANGUAGE_STATE_KEY] = DEFAULT_LANGUAGE

    if TIMEZONE_STATE_KEY not in st.session_state:
        st.session_state[TIMEZONE_STATE_KEY] = DEFAULT_PRESENTATION_TIMEZONE


def render_presentation_preferences() -> tuple[LanguageCode, str]:
    """Render global display preferences without changing analytical data."""
    _initialize_presentation_preferences()
    current_language = str(st.session_state[LANGUAGE_STATE_KEY])
    language = (
        cast(LanguageCode, current_language)
        if current_language in SUPPORTED_LANGUAGES
        else DEFAULT_LANGUAGE
    )

    st.sidebar.markdown(f"### {translate('preferences.heading', language)}")
    selected_language = st.sidebar.selectbox(
        translate("preferences.language", language),
        options=SUPPORTED_LANGUAGES,
        format_func=lambda item: LANGUAGE_NAMES[item],
        key=LANGUAGE_STATE_KEY,
    )
    selected_timezone = st.sidebar.selectbox(
        translate("preferences.timezone", selected_language),
        options=SUPPORTED_PRESENTATION_TIMEZONES,
        key=TIMEZONE_STATE_KEY,
    )
    st.sidebar.divider()

    return selected_language, selected_timezone


def _select_page(
    page_name: str,
) -> None:
    """Update selected application page."""
    st.session_state[SELECTED_PAGE_STATE_KEY] = page_name


def _navigation_key(
    page_name: str,
) -> str:
    """Build deterministic Streamlit widget key."""
    normalized = (
        page_name.lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "&",
            "and",
        )
    )

    return "navigation_" + normalized


def render_module_navigation(
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> str:
    """Render button-based module navigation."""
    _initialize_selected_page()

    selected_page = str(st.session_state[SELECTED_PAGE_STATE_KEY])

    st.sidebar.markdown(f"### {translate('navigation.heading', language)}")

    st.sidebar.caption(translate("navigation.caption", language))

    for page_name in NAVIGATION_OPTIONS:
        is_selected = page_name == selected_page

        st.sidebar.button(
            translate(NAVIGATION_TRANSLATION_KEYS[page_name], language),
            key=_navigation_key(page_name),
            type=("primary" if is_selected else "secondary"),
            icon=(NAVIGATION_ICONS.get(page_name)),
            width=270,
            on_click=_select_page,
            args=(page_name,),
        )

    return selected_page


def render_system_status(
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Render backend status in sidebar."""
    st.sidebar.divider()

    st.sidebar.markdown(f"### {translate('system.heading', language)}")

    (
        api_available,
        health,
        information,
        error_message,
    ) = load_api_status(client)

    if api_available:
        st.sidebar.success(translate("system.api_available", language))

        if health is not None:
            health_status = str(
                health.get(
                    "status",
                    "unknown",
                )
            )

            st.sidebar.caption(
                translate("system.backend_status", language, status=health_status)
            )

        if information is not None:
            api_version = information.get("version")

            if api_version is not None:
                st.sidebar.caption(
                    translate("system.api_version", language, version=api_version)
                )
    else:
        st.sidebar.error(translate("system.api_unavailable", language))

        if error_message:
            st.sidebar.caption(error_message)

    st.sidebar.caption(translate("system.backend", language, url=client.base_url))

    st.sidebar.divider()

    st.sidebar.caption(translate("system.http_boundary", language))


def render_sidebar(
    client: APIClient,
) -> tuple[str, LanguageCode, str]:
    """Render application sidebar."""
    language, presentation_timezone = render_presentation_preferences()
    selected_page = render_module_navigation(language)

    render_system_status(client, language)

    return selected_page, language, presentation_timezone


def render_selected_page(
    selected_page: str,
    client: APIClient,
    language: LanguageCode = DEFAULT_LANGUAGE,
    presentation_timezone: str = DEFAULT_PRESENTATION_TIMEZONE,
) -> None:
    """Render the selected application module."""
    if selected_page == ABOUT_PAGE:
        render_about_page(language=language)
        return

    if selected_page == DASHBOARD_PAGE:
        render_dashboard_page(client, language=language)
        return

    if selected_page == FORECASTING_PAGE:
        render_forecast_page(client, language=language)
        return

    if selected_page == RISK_SCENARIOS_PAGE:
        render_risk_page(client, language=language)
        return

    if selected_page == OPTIMIZATION_PAGE:
        render_optimization_page(client, language=language)
        return

    if selected_page == DECISION_WORKSPACE_PAGE:
        render_decision_workspace_page(
            client,
            language=language,
            presentation_timezone=presentation_timezone,
        )
        return

    if selected_page == MARKET_INTELLIGENCE_PAGE:
        render_market_intelligence_page(
            client,
            language=language,
            presentation_timezone=presentation_timezone,
        )
        return

    if selected_page == AI_INSIGHTS_PAGE:
        render_ai_insights_page(client, language=language)
        return

    if selected_page == GOVERNANCE_PAGE:
        render_governance_page(client, language=language)
        return

    if selected_page == REPORTS_PAGE:
        render_reports_page(client, language=language)
        return

    st.error(translate("page.unknown", language))


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    render_branding_styles()
    render_accessibility_styles()

    client = create_api_client()

    selected_page, language, presentation_timezone = render_sidebar(client)

    render_selected_page(
        selected_page,
        client,
        language,
        presentation_timezone,
    )


if __name__ == "__main__":
    main()

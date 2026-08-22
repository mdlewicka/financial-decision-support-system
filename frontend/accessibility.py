"""Small accessibility helpers shared by the Streamlit presentation layer."""

from typing import Final

import streamlit as st

from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate

ACCESSIBILITY_STYLES: Final = """
<style>
:where(a, button, input, select, textarea, [tabindex]):focus-visible {
    outline: 3px solid #005fcc !important;
    outline-offset: 3px !important;
    border-radius: 0.25rem;
}

a {
    text-decoration-thickness: 0.12em;
    text-underline-offset: 0.18em;
}

[data-testid="stAlert"] {
    border: 2px solid currentColor;
}

@media (prefers-color-scheme: dark) {
    :where(a, button, input, select, textarea, [tabindex]):focus-visible {
        outline-color: #ffbf47 !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""


def render_accessibility_styles() -> None:
    """Install deterministic focus, contrast and reduced-motion styles."""
    st.markdown(
        ACCESSIBILITY_STYLES,
        unsafe_allow_html=True,
    )


def render_chart_description(
    description: str,
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Provide a visible text alternative next to an interactive chart."""
    st.caption(
        translate(
            "accessibility.chart_description",
            language,
            description=description,
        )
    )

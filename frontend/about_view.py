"""Introductory application guide for first-time users."""

import streamlit as st

from frontend.i18n import DEFAULT_LANGUAGE, LanguageCode, translate


def render_about_page(
    language: LanguageCode = DEFAULT_LANGUAGE,
) -> None:
    """Explain the system, its data and the purpose of every module."""
    st.title(translate("about.title", language))
    st.caption(translate("about.caption", language))

    st.info(translate("about.disclaimer", language))

    purpose_column, data_column, results_column = st.columns(3)

    with purpose_column:
        st.subheader(translate("about.purpose.title", language))
        st.markdown(translate("about.purpose.body", language))

    with data_column:
        st.subheader(translate("about.data.title", language))
        st.markdown(translate("about.data.body", language))

    with results_column:
        st.subheader(translate("about.results.title", language))
        st.markdown(translate("about.results.body", language))

    st.divider()
    st.header(translate("about.modules.title", language))
    st.caption(translate("about.modules.caption", language))

    module_keys = (
        "dashboard",
        "forecasting",
        "risk",
        "optimization",
        "decision",
        "market",
        "ai",
        "governance",
        "reports",
    )

    for row_start in range(0, len(module_keys), 3):
        columns = st.columns(3)
        for column, module_key in zip(
            columns,
            module_keys[row_start : row_start + 3],
            strict=True,
        ):
            with column:
                st.subheader(translate(f"about.module.{module_key}.title", language))
                st.markdown(translate(f"about.module.{module_key}.body", language))

    st.divider()
    st.subheader(translate("about.start.title", language))
    st.markdown(translate("about.start.body", language))

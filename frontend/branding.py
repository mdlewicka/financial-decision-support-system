"""Corporate visual identity for the Streamlit application."""

from typing import Final

import streamlit as st

CORPORATE_NAVY: Final = "#163A5F"
CORPORATE_NAVY_DARK: Final = "#0B1F33"
CORPORATE_BLUE: Final = "#2F6B9A"

CORPORATE_STYLE: Final = f"""
<style>
    :root {{
        --fdss-navy: {CORPORATE_NAVY};
        --fdss-navy-dark: {CORPORATE_NAVY_DARK};
        --fdss-blue: {CORPORATE_BLUE};
        --fdss-surface: #FFFFFF;
        --fdss-background: #FBFCFE;
        --fdss-border: #DCE5EF;
        --fdss-text: #152238;
        --fdss-muted: #52657A;
    }}

    [data-testid="stAppViewContainer"] {{
        background:
            linear-gradient(180deg, #F7FAFD 0, #FBFCFE 13rem, #FBFCFE 100%);
        color: var(--fdss-text);
    }}

    [data-testid="stHeader"] {{
        background: rgba(251, 252, 254, 0.94);
        border-bottom: 1px solid var(--fdss-border);
    }}

    [data-testid="stSidebar"] {{
        border-right: 1px solid #365875;
    }}

    [data-testid="stSidebar"] h3 {{
        color: #FFFFFF;
        letter-spacing: 0.015em;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: #D7E3EF;
    }}

    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3 {{
        color: var(--fdss-navy-dark);
    }}

    [data-testid="stMainBlockContainer"] h1 {{
        letter-spacing: -0.025em;
    }}

    [data-testid="stMainBlockContainer"] h4 {{
        color: var(--fdss-navy-dark);
        font-weight: 700;
        margin-top: 1.15rem;
        margin-bottom: 0.8rem;
    }}

    [data-testid="stMetric"] {{
        min-height: 7.4rem;
        padding: 1rem 1.1rem;
        background: var(--fdss-surface);
        border: 1px solid var(--fdss-border);
        border-top: 0.25rem solid var(--fdss-navy);
        border-radius: 0.8rem;
        box-shadow: 0 0.25rem 0.9rem rgba(11, 31, 51, 0.06);
        text-align: center;
    }}

    [data-testid="stMetric"] > div {{
        width: 100%;
        text-align: center;
    }}

    [data-testid="stMetricLabel"] {{
        width: 100% !important;
        display: flex !important;
        grid-template-columns: none !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.25rem;
        color: var(--fdss-muted);
        font-weight: 650;
        text-align: center !important;
    }}

    [data-testid="stMetricLabel"] > div {{
        width: auto !important;
        flex: 0 1 auto !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }}

    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] {{
        width: auto !important;
        display: block !important;
        text-align: center !important;
    }}

    [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span {{
        width: auto !important;
        margin: 0 auto !important;
        text-align: center !important;
    }}

    [data-testid="stMetricValue"] {{
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        color: var(--fdss-navy-dark);
        text-align: center !important;
    }}

    [data-testid="stMetricValue"] > div {{
        width: auto !important;
        flex: 0 1 auto !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center !important;
    }}

    [data-testid="stMetricValue"] p,
    [data-testid="stMetricValue"] span {{
        width: auto !important;
        margin: 0 auto !important;
        text-align: center !important;
    }}

    [data-testid="stDataFrame"] {{
        background: var(--fdss-surface);
        border: 1px solid #C7D4E2;
        border-top: 0.2rem solid var(--fdss-navy);
        border-radius: 0.85rem;
        box-shadow: 0 0.25rem 0.85rem rgba(11, 31, 51, 0.07);
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        overflow: hidden;
    }}

    [data-testid="stDataFrame"] > div {{
        background: var(--fdss-surface);
    }}

    [data-testid="stDataFrame"]:focus-within {{
        border-color: #8FA8C0;
        border-top-color: var(--fdss-navy-dark);
        box-shadow: 0 0.3rem 0.95rem rgba(11, 31, 51, 0.09);
    }}

    .fdss-table-wrapper {{
        max-width: 100%;
        margin-top: 0.45rem;
        margin-bottom: 0.9rem;
        overflow-x: auto;
        background: var(--fdss-surface);
        border: 1px solid #C7D4E2;
        border-top: 0.2rem solid var(--fdss-navy);
        border-radius: 0.85rem;
        box-shadow: 0 0.25rem 0.85rem rgba(11, 31, 51, 0.07);
    }}

    .fdss-table {{
        width: 100%;
        margin: 0;
        border: 0;
        border-collapse: collapse;
        border-spacing: 0;
        background: var(--fdss-surface);
        color: var(--fdss-text);
        font-size: 0.93rem;
    }}

    .fdss-table thead th {{
        padding: 0.72rem 0.8rem;
        text-align: center !important;
        vertical-align: middle !important;
        background: #F3F6FA;
        color: var(--fdss-muted);
        font-weight: 650;
        white-space: nowrap;
        border-right: 1px solid #E2E9F1;
        border-bottom: 1px solid #DCE5EF;
    }}

    .fdss-table tbody td,
    .fdss-table tbody th {{
        padding: 0.72rem 0.8rem;
        text-align: center !important;
        vertical-align: middle !important;
        background: var(--fdss-surface);
        color: var(--fdss-text);
        white-space: nowrap;
        border-right: 1px solid #E8EDF3;
        border-bottom: 1px solid #E8EDF3;
    }}

    .fdss-table tbody tr:nth-child(even) td,
    .fdss-table tbody tr:nth-child(even) th {{
        background: #FBFCFE;
    }}

    .fdss-table tbody tr:hover td,
    .fdss-table tbody tr:hover th {{
        background: #F5F8FC;
    }}

    .fdss-table thead th:last-child,
    .fdss-table tbody td:last-child,
    .fdss-table tbody th:last-child {{
        border-right: 0;
    }}

    .fdss-table tbody tr:last-child td,
    .fdss-table tbody tr:last-child th {{
        border-bottom: 0;
    }}

    [data-testid="stExpander"] {{
        background: var(--fdss-surface);
        border-radius: 0.75rem;
    }}

    .stButton > button[kind="primary"] {{
        background: var(--fdss-navy);
        border-color: var(--fdss-navy);
        color: #FFFFFF;
        font-weight: 650;
    }}

    .stButton > button[kind="primary"]:hover {{
        background: var(--fdss-navy-dark);
        border-color: var(--fdss-navy-dark);
    }}

    [data-testid="stMainBlockContainer"] .stButton > button[kind="secondary"] {{
        border-color: #9EB2C6;
        color: var(--fdss-navy-dark);
    }}

    [data-testid="stAlert"] {{
        border-radius: 0.7rem;
    }}
</style>
"""


def render_branding_styles() -> None:
    """Apply the presentation-only corporate visual identity."""
    st.markdown(
        CORPORATE_STYLE,
        unsafe_allow_html=True,
    )

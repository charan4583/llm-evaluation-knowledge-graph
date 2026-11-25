"""Dark-first developer-tool theme and reusable UI components for Streamlit."""

from __future__ import annotations

import streamlit as st

DARK_THEME_CSS = """
<style>
    /* Global background and typography */
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
        letter-spacing: -0.015em;
        font-weight: 600;
    }
    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #111c33;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 600;
    }
    /* Custom card container */
    .app-card {
        background-color: #111c33;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .app-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 6px;
    }
    .app-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid #334155;
        background-color: #1e293b;
        color: #cbd5e1;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.14);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }
    .badge-warn {
        background-color: rgba(245, 158, 11, 0.14);
        border-color: rgba(245, 158, 11, 0.4);
        color: #fbbf24;
    }
    .badge-danger {
        background-color: rgba(239, 68, 68, 0.14);
        border-color: rgba(239, 68, 68, 0.4);
        color: #f87171;
    }
    .badge-info {
        background-color: rgba(56, 189, 248, 0.14);
        border-color: rgba(56, 189, 248, 0.4);
        color: #38bdf8;
    }
    /* Clean buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
</style>
"""

LIGHT_THEME_CSS = """
<style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid #cbd5e1;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .app-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    .app-card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .app-muted {
        color: #475569;
        font-size: 0.88rem;
    }
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid #cbd5e1;
        background-color: #e2e8f0;
        color: #1e293b;
        margin-right: 6px;
        margin-bottom: 4px;
    }
</style>
"""


def apply_theme() -> None:
    """Inject the selected theme CSS and sidebar theme toggle."""
    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "dark"

    with st.sidebar:
        st.caption("Appearance")
        theme_choice = st.radio(
            "Theme",
            options=["dark", "light"],
            index=0 if st.session_state["ui_theme"] == "dark" else 1,
            format_func=lambda x: "Dark (Default)" if x == "dark" else "Light",
            horizontal=True,
            label_visibility="collapsed",
            key="theme_radio",
        )
        st.session_state["ui_theme"] = theme_choice

    css = DARK_THEME_CSS if st.session_state["ui_theme"] == "dark" else LIGHT_THEME_CSS
    st.markdown(css, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str) -> None:
    """Render a consistent page title and subtitle."""
    st.title(title)
    st.caption(subtitle)

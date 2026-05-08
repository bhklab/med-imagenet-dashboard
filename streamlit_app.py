"""
Med-ImageNet Explorer

Multi-page Streamlit app for browsing, visualising, and exporting metadata
across indexed medical imaging datasets.
"""

import streamlit as st

st.set_page_config(
    page_title="Med-ImageNet Explorer",
    page_icon=":hospital:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- global styling -----------------------------------------------------------

st.markdown(
    """
    <style>
    /* Metric cards -- theme-aware via CSS custom properties */
    [data-testid="stMetric"] {
        background: color-mix(in srgb, var(--background-color, transparent) 85%, var(--text-color, gray) 15%);
        border: 1px solid color-mix(in srgb, var(--text-color, gray) 10%, transparent 90%);
        border-radius: .5rem;
        padding: .75rem 1rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .04em;
        opacity: .7;
    }
    [data-testid="stMetricValue"] {
        font-weight: 700;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Softer container borders */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stContainer"]) {
        border-radius: .6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -- navigation ---------------------------------------------------------------

home_page = st.Page(
    "pages/home.py",
    title="Home",
    icon=":material/home:",
    url_path="home",
    default=True,
)
datasets_page = st.Page(
    "pages/datasets.py",
    title="Collections",
    icon=":material/dataset:",
    url_path="collections",
)

pg = st.navigation(
    {
        "": [home_page],
        "Explore": [datasets_page],
        # metrics_page],
        # "AI": [chat_page],
    }
)

pg.run()

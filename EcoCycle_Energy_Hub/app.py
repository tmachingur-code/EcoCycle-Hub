"""EcoCycle Energy Hub — modular Streamlit application.

Launch from the project root with: streamlit run app.py
"""
import streamlit as st
from components.styles import inject_styles
from components.layout import topbar, navigation, footer
from core.history import initialise_history
from pages import home, scanner, dashboard, impact, how_it_works, sorting_system, about

st.set_page_config(
    page_title="EcoCycle Energy Hub | Smart Material Intelligence",
    page_icon="♻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PAGES = {
    "Home": home.render,
    "AI Scanner": scanner.render,
    "Dashboard": dashboard.render,
    "Impact": impact.render,
    "How it works": how_it_works.render,
    "Smart Sorting System": sorting_system.render,
    "About": about.render,
}


def main():
    inject_styles()
    initialise_history()
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    topbar()
    navigation()
    PAGES.get(st.session_state.page, home.render)()
    footer()


if __name__ == "__main__":
    main()

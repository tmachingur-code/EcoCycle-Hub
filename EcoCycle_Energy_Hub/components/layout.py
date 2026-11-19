import streamlit as st
from config.settings import NAV_ITEMS

def topbar():
    st.markdown('''<div class="topbar"><div class="brand"><div class="brand-mark">E</div><div>EcoCycle Energy Hub<small>SMART MATERIAL INTELLIGENCE</small></div></div><div class="status-pill">● AI PROTOTYPE ONLINE</div></div>''', unsafe_allow_html=True)

def navigation():
    current = st.session_state.get("page", "Home")
    st.markdown('<div class="nav-wrap">', unsafe_allow_html=True)
    cols = st.columns(len(NAV_ITEMS))
    for col, item in zip(cols, NAV_ITEMS):
        with col:
            if st.button(item, key=f"nav_{item}", type="primary" if current == item else "secondary", use_container_width=True):
                st.session_state.page = item
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def page_heading(title, subtitle):
    st.markdown(f'<div class="page-title">{title}</div><p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-kicker">{label}</div>', unsafe_allow_html=True)

def footer():
    st.markdown('<div class="footer">EcoCycle Energy Hub · Turning waste intelligence into circular opportunities<br>Image-based detection and classification prototype</div>', unsafe_allow_html=True)

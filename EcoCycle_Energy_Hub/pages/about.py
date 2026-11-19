import streamlit as st
from components.layout import page_heading, section

def render():
    page_heading("About EcoCycle", "Technology for practical, inclusive and data-informed circular economy solutions.")
    st.markdown('<div class="hero" style="min-height:220px"><div class="hero-copy"><div class="eyebrow">OUR PURPOSE</div><h1>Waste is not<br>the end of value.</h1><p>EcoCycle Energy Hub explores how accessible technology can support the identification, recovery and productive use of recyclable materials.</p></div></div>',unsafe_allow_html=True)
    section("Mission")
    st.markdown('<div class="feature-panel"><h3>Make material recovery more intelligent and accessible.</h3><p>We aim to develop practical tools that help communities and waste-sector actors identify valuable material streams and strengthen circular economy opportunities.</p></div>',unsafe_allow_html=True)
    section("Our principles")
    cols=st.columns(3)
    for col,title,body in zip(cols,["Practical innovation","Inclusive access","Evidence-led impact"],["Build useful technology around real operational needs.","Consider the needs of collectors, sorters and communities.","Measure outcomes rather than making unsupported impact claims."]):
        with col: st.markdown(f'<div class="panel" style="height:100%"><h3>{title}</h3><p style="color:#6b7b70">{body}</p></div>',unsafe_allow_html=True)
    st.caption("EcoCycle Energy Hub · Software prototype")

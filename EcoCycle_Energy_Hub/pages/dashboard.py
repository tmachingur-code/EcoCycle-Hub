import streamlit as st
from components.layout import page_heading, section
from core.history import history_frame

def render():
    page_heading("Scan Dashboard", "A live overview of the scans recorded during this app session.")
    df = history_frame()
    total = len(df)
    a,b,c,d = st.columns(4)
    a.metric("Objects analysed", total)
    b.metric("PET", int((df.Material == "PET").sum()) if total else 0)
    c.metric("HDPEM", int((df.Material == "HDPEM").sum()) if total else 0)
    d.metric("OTHER", int((df.Material == "OTHER").sum()) if total else 0)
    section("Material distribution")
    if total:
        counts = df.Material.value_counts().reindex(["PET","HDPEM","OTHER"], fill_value=0)
        st.bar_chart(counts)
        section("Recent scan activity")
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Download scan history (CSV)", df.to_csv(index=False).encode("utf-8"), "ecocycle_scan_history.csv", "text/csv")
        if st.button("Clear session history", type="secondary"):
            st.session_state.ecocycle_history = []; st.rerun()
    else:
        st.markdown('<div class="feature-panel"><h3>Your dashboard is ready</h3><p>Once you analyse an image in the AI Scanner, detected objects and material predictions will appear here.</p></div>', unsafe_allow_html=True)

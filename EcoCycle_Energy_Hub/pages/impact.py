
import streamlit as st

from components.layout import page_heading, section
from config.settings import ASSET_DIR


def render():

    page_heading(
        "Impact & Opportunity",
        "The role of material intelligence in a more circular waste system."
    )

    # HERO
    st.markdown(
        """
        <div class="hero" style="min-height:210px">
            <div class="hero-copy">
                <div class="eyebrow">FROM DISPOSAL TO RESOURCE</div>
                <h1>Better information.<br>Better recovery.</h1>
                <p>
                    Identifying material types is one step towards improving
                    sorting decisions, supporting recovery pathways and
                    reducing contamination.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # WHERE THIS TECHNOLOGY CAN CONTRIBUTE
    section("Where this technology can contribute")

    cols = st.columns(3)

    cards = [
        (
            "Material visibility",
            "Make material categories more visible through image-based analysis."
        ),
        (
            "Sorting support",
            "Provide preliminary classification insights that can support human sorting workflows."
        ),
        (
            "Circular value",
            "Help organisations understand what recyclable streams may be present in collected waste."
        )
    ]

    for col, (title, desc) in zip(cols, cards):

        with col:

            st.markdown(
                f"""
                <div class="panel" style="height:100%">
                    <h3>{title}</h3>
                    <p style="color:#6b7b70">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ENVIRONMENTAL IMPACT VISUAL
    section("A more sustainable future")

    impact_image = ASSET_DIR / "ecocycle_impact.png"

    if impact_image.exists():

        st.image(
            str(impact_image),
            use_container_width=True
        )

    else:

        st.info(
            "Impact image not found. Add ecocycle_impact.png to your assets folder."
        )

    
    # IMPACT MEASUREMENT ROADMAP
    section("Impact measurement roadmap")

    st.html("""
        <div class="panel">

            <div class="mini-label">PHASE 1</div>
            <h3>Prototype stage</h3>
            <p>
                Measure classification performance, confidence calibration
                and the frequency of uncertain predictions.
            </p>

            <hr>

            <div class="mini-label">PHASE 2</div>
            <h3>Pilot stage</h3>
            <p>
                With a real collection partner, measure sorting time,
                contamination rates, recovered PET/HDPE volumes
                and operator feedback.
            </p>

            <hr>

            <div class="mini-label">PHASE 3</div>
            <h3>Deployment stage</h3>
            <p>
                Track recovery rates, operating costs, participation
                and verified environmental outcomes over time.
            </p>

        </div>
    """)

    st.caption(
        "No environmental impact totals are claimed by this prototype. "
        "Impact figures should be reported only after field measurement and verification."
    )

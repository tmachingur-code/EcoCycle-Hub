
import streamlit as st
from textwrap import dedent

from config.settings import ASSET_DIR
from components.layout import section

def html(content):
    """Render raw HTML directly in Streamlit."""
    st.html(content)

def render():

    # HERO — ECOCYCLE IDENTITY
    html("""
        <div class="hero">
            <div class="hero-copy">
                <div class="eyebrow">
                    ECOCYCLE ENERGY HUB
                </div>

                <h1>
                    Turning waste<br>
                    into valuable resources.
                </h1>

                <p>
                    Rethinking waste management through innovation,
                    technology and circular economy solutions.
                </p>

                <div style="margin-top:24px;">
                    <span class="status-pill">
                        BUILDING A CIRCULAR FUTURE
                    </span>
                </div>
            </div>

            <div class="hero-aside">
                <div class="orbit">
                    <span>ECO<br>AI</span>
                </div>
            </div>
        </div>
    """)

    # OUR MISSION
    section("WHO WE ARE")

    left, right = st.columns([1, 1], gap="large")

    with left:
        html("""
            <h2 style="margin-top:0;">
                Waste is not the end of a product's journey.
            </h2>

            <p style="color:#6b7b70;">
                EcoCycle Energy Hub is an innovation initiative
                exploring how technology can transform plastic
                waste into valuable resources.
            </p>

            <p style="color:#6b7b70;">
                We aim to support smarter waste identification,
                improved recovery pathways and a more sustainable
                approach to resource management.
            </p>
        """)

    with right:
        image = ASSET_DIR /  "home_hero.jpg"

        if image.exists():
            st.image(
                str(image),
                use_container_width=True,
            )

    # THE CHALLENGE
    section("THE CHALLENGE")

    html("""
        <div class="feature-panel">
            <h2>From waste accumulation to resource recovery.</h2>

            <p>
                Plastic waste presents environmental and resource
                management challenges. When recyclable materials
                are mixed, poorly identified or improperly handled,
                opportunities for recovery are lost.
            </p>

            <p>
                EcoCycle is exploring technology-driven approaches
                that can help make material identification and
                recycling decisions more informed.
            </p>
        </div>
    """)

    # OUR VISION
    section("OUR VISION")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        image = ASSET_DIR / "smarter_resource_recovery.png"

        if image.exists():
            st.image(
                str(image),
                use_container_width=True,
            )

        st.markdown("""
            ### Smarter resource recovery

            Building towards systems that can support
            more efficient identification and recovery
            of recyclable materials.
        """)

    with col2:
        image = ASSET_DIR / "circular_economy.png"

        if image.exists():
            st.image(
                str(image),
                use_container_width=True,
            )

        st.markdown("""
            ### A circular economy

            Exploring opportunities to keep valuable
            materials in use for longer and reduce
            unnecessary waste.
        """)

    # CURRENT INITIATIVE
    section("WHERE WE ARE TODAY")

    html("""
        <div class="panel">
            <div class="mini-label">
                OUR CURRENT DEVELOPMENT
            </div>

            <h2>Building the EcoCycle AI prototype</h2>

            <p style="color:#6b7b70;">
                We have developed an image-classification
                application that uses MobileNetV2 to identify
                material categories from uploaded images.
                This is an early step towards our broader
                vision for intelligent waste management.
            </p>
        </div>
    """)

    # EXPLORE
    section("EXPLORE ECOCYCLE")

    html("""
        <div class="feature-panel">
            <h2>Experience our current prototype.</h2>

            <p>
                Explore the AI Scanner and see how EcoCycle
                analyses material images.
            </p>
        </div>
    """)

    if st.button(
        "Explore the AI Scanner →",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.page = "AI Scanner"
        st.rerun()

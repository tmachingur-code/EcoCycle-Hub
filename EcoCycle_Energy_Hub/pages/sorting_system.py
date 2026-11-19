
import streamlit as st

from components.layout import page_heading, section
from config.settings import ASSET_DIR


def render():

    page_heading(
        "Smart Sorting System",
        "The current software prototype and the longer-term physical sorting vision."
    )

    # HERO
    st.html("""
        <div class="hero" style="min-height:205px">
            <div class="hero-copy">
                <div class="eyebrow">CURRENT BUILD</div>
                <h1>Classify. Understand.<br>Inform decisions.</h1>
                <p>
                    EcoCycle currently analyses uploaded images
                    and presents material predictions for human review.
                </p>
            </div>
        </div>
    """)

    # CURRENT SORTING INFRASTRUCTURE
    section("SMART SORTING INFRASTRUCTURE")

    facility = ASSET_DIR / "sorting_facility.jpg"

    if facility.exists():

        st.image(
            str(facility),
            use_container_width=True,
            caption="Illustrative sorting facility · Future deployment context"
        )

    # CURRENT PROTOTYPE CAPABILITIES
    section("WHAT IS WORKING IN THE PROTOTYPE")

    capabilities = [
        (
            "Image input",
            "Upload a photo through the Streamlit interface."
        ),
        (
            "AI inference",
            "MobileNetV2 analyses the complete uploaded image and predicts PET, HDPEM or OTHER."
        ),
        (
            "Review & records",
            "View prediction confidence, material probabilities and session-level scan history."
        ),
    ]

    columns = st.columns(3, gap="medium")

    for col, (title, description) in zip(columns, capabilities):

        with col:

            st.html(f"""
                <div class="panel" style="height:100%; min-height:175px;">
                    <h3>{title}</h3>
                    <p style="color:#6b7b70;">
                        {description}
                    </p>
                </div>
            """)

    # FUTURE DEVELOPMENT PATHWAY
    section("FUTURE DEVELOPMENT PATHWAY")

    st.html("""
        <div class="panel">

            <div class="mini-label">
                PHASE 1 · SOFTWARE VALIDATION
            </div>

            <h3>Improve model reliability</h3>

            <p>
                Expand representative training data, review class-level
                performance, evaluate low-confidence predictions and
                test under real operating conditions.
            </p>

            <hr>

            <div class="mini-label">
                PHASE 2 · OPERATOR WORKFLOW
            </div>

            <h3>Connect to a sorting station</h3>

            <p>
                Design a safe operator interface and validate camera
                placement, lighting, throughput and material presentation.
            </p>

            <hr>

            <div class="mini-label">
                PHASE 3 · PHYSICAL AUTOMATION
            </div>

            <h3>Prototype mechanical separation</h3>

            <p>
                Only after feasibility and safety testing, explore
                actuators or conveyor integration for physical separation.
            </p>

        </div>
    """)

    st.html("""
        <div class="soft-note">
            <strong>Important:</strong>
            Conveyor control, robotic actuation and physical material
            separation are not implemented in the current application.
            YOLO11 is a planned future development, not part of the
            current scanning pipeline.
        </div>
    """)


import hashlib
import streamlit as st
import pandas as pd
from PIL import Image

from components.layout import page_heading, section
from core.model_loader import load_classifier
from core.classifier import classify_crop
from core.history import add_scan
from core.image_processing import prepare_image

from config.settings import (
    ASSET_DIR,
    CLASS_DESCRIPTIONS,
    CLASS_GUIDANCE,
    CLASS_NAMES,
)


def render():

    page_heading(
        "AI Material Scanner",
        "Upload a waste image and let EcoCycle's MobileNetV2 AI classify its material."
    )

    left, right = st.columns([1.05, 0.95], gap="large")

    # IMAGE UPLOAD
    with left:

        section("01 · Upload your waste image")

        uploaded = st.file_uploader(
            "Choose a JPG, JPEG or PNG image",
            type=["jpg", "jpeg", "png"],
            key="scanner_upload"
        )

        run = st.button(
            "Analyse material",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None
        )

    # SHOW EXAMPLE IMAGE BEFORE UPLOAD
    if uploaded is None:

        with right:

            section("Example AI classification")

            demo_image = ASSET_DIR / "ai_scanner_demo.jpg"

            if demo_image.exists():

                st.image(
                    str(demo_image),
                    use_container_width=True
                )

                st.caption(
                    "Illustrative MobileNetV2 classification result. "
                    "Actual predictions depend on the uploaded image."
                )

            else:

                st.info(
                    "Example image not found. "
                    "Please check your assets folder."
                )

    # PROCESS UPLOADED IMAGE
    if uploaded is not None:

        try:

            image = Image.open(uploaded)
            image = prepare_image(image)

            digest = hashlib.sha256(
                uploaded.getvalue()
            ).hexdigest()

            # SHOW ACTUAL UPLOADED IMAGE
            with right:

                section("Image preview")

                st.image(
                    image,
                    use_container_width=True
                )

            # RUN CLASSIFICATION
            if run:

                try:

                    classifier = load_classifier()

                    with st.spinner(
                        "Analysing material with MobileNetV2..."
                    ):

                        result = classify_crop(
                            image,
                            classifier
                        )

                    st.session_state.scan_result = {
                        "digest": digest,
                        "name": uploaded.name,
                        "image": image,
                        "result": result,
                    }

                    # SAVE EACH IMAGE ONLY ONCE PER SESSION
                    if st.session_state.get(
                        "saved_scan_digest"
                    ) != digest:

                        add_scan(
                            uploaded.name,
                            result
                        )

                        st.session_state.saved_scan_digest = digest

                except Exception as exc:

                    st.error(
                        f"Classification failed: {exc}"
                    )

                    st.info(
                        "Check your MobileNetV2 model file and dependencies."
                    )

        except Exception as exc:

            st.error(
                f"Could not open this image: {exc}"
            )

    # DISPLAY CLASSIFICATION RESULTS
    result_data = st.session_state.get("scan_result")

    if (
        result_data
        and uploaded is not None
        and result_data["digest"] == hashlib.sha256(
            uploaded.getvalue()
        ).hexdigest()
    ):

        prediction = result_data["result"]

        material = prediction["material"]
        confidence = prediction["confidence"]
        probabilities = prediction["probabilities"]

        section("02 · AI classification results")

        st.markdown(
            f"""
            <div class="feature-panel">
                <div class="mini-label">Predicted material</div>
                <div class="page-title">{material}</div>
                <p>{CLASS_DESCRIPTIONS[material]}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction confidence",
                f"{confidence:.2%}"
            )

        with col2:

            st.metric(
                "Model",
                "MobileNetV2"
            )

        # PROBABILITY BREAKDOWN
        section("03 · Material probability breakdown")

        probability_df = pd.DataFrame({
            "Material": CLASS_NAMES,
            "Probability": [
                probabilities[name]
                for name in CLASS_NAMES
            ]
        })

        st.bar_chart(
            probability_df.set_index("Material"),
            y="Probability",
            use_container_width=True
        )

        for name in CLASS_NAMES:

            value = probabilities[name]

            st.progress(
                value,
                text=f"{name} · {value:.2%}"
            )

        # SORTING RECOMMENDATION
        section("04 · Sorting recommendation")

        if confidence < 0.60:

            st.warning(
                "The model has relatively low confidence. "
                "Please review this material manually before sorting."
            )

        else:

            st.success(
                CLASS_GUIDANCE[material]
            )

        st.caption(
            "Classification applies to the complete uploaded image. "
            "This version does not detect separate objects within an image."
        )

    elif result_data and uploaded is not None:

        st.info(
            "You have selected a different image. "
            "Click Analyse material to classify it."
        )

    # CURRENT SYSTEM LIMITATIONS
    st.markdown(
        """
        <div class="soft-note">
            EcoCycle is currently operating in image-classification mode.
            Physical sorting, conveyor control and automated object detection
            are not active in this version.
        </div>
        """,
        unsafe_allow_html=True
    )

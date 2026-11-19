
import streamlit as st
from textwrap import dedent

from components.layout import page_heading, section
from config.settings import ASSET_DIR, TEST_ACCURACY, TEST_IMAGE_COUNT


def render():

    page_heading(
        "How It Works",
        "Discover how EcoCycle uses artificial intelligence to identify "
        "plastic materials from uploaded images."
    )

    # 01. CLASSIFICATION PROCESS
    section("THE CLASSIFICATION PROCESS")

    stages = [
        (
            "Upload an image",
            "Upload a clear JPG, JPEG or PNG image of the material you want to identify.",
            ["step_upload.png", "01_upload_an_image.png"]
        ),
        (
            "Image preprocessing",
            "The image is converted to RGB, resized to 224 × 224 pixels and prepared for the neural network.",
            ["step_preprocessing.png", "02_image_preprocessing.png"]
        ),
        (
            "AI material classification",
            "MobileNetV2 analyses the complete image and estimates its PET, HDPEM or OTHER probability.",
            ["step_classification.png", "03_ai_material_classification.png"]
        ),
        (
            "Prediction and confidence",
            "EcoCycle displays the predicted material, confidence score and probability distribution.",
            ["step_prediction.png", "04_prediction_and_confidence.png"]
        ),
        (
            "Review and record",
            "Review the recommendation and explore your classification history in the dashboard.",
            ["step_review.png", "05_review_and_record.png"]
        ),
    ]

    
    for index, (title, description, filenames) in enumerate(stages, start=1):

        image_path = next(
            (
                ASSET_DIR / filename
                for filename in filenames
                if (ASSET_DIR / filename).exists()
            ),
            None
        )

        with st.container(border=True):

            image_col, text_col = st.columns([1, 3], gap="medium")

            with image_col:
                if image_path:
                    st.image(
                        str(image_path),
                        width=160
                    )
                else:
                    st.warning(f"Image {index} not found in assets.")

            with text_col:
                st.markdown(f"**{index:02d} · {title}**")
                st.write(description)


    # 02. MODEL EXPLANATION
    section("INSIDE THE AI MODEL")

    st.html("""
        <div class="panel">
            <div class="mini-label">DEEP LEARNING CLASSIFIER</div>

            <h2>MobileNetV2</h2>

            <p style="color:#6b7b70;">
                MobileNetV2 is a convolutional neural network
                architecture used by EcoCycle to analyse uploaded
                images and predict their material categories.
                The model produces probabilities for three classes.
            </p>
        </div>
    """)

    st.write("")

    # 03. MATERIAL CATEGORIES
    section("MATERIAL CATEGORIES")

    materials = [
        (
            "PET",
            "Polyethylene Terephthalate",
            "Commonly used in beverage bottles and plastic packaging."
        ),
        (
            "HDPEM",
            "High-Density Polyethylene",
            "Commonly used in milk containers and rigid packaging."
        ),
        (
            "OTHER",
            "Other trained material categories",
            "Materials outside the PET and HDPEM classes."
        ),
    ]

    columns = st.columns(3, gap="medium")

    for column, (name, full_name, description) in zip(columns, materials):

        with column:

            st.html(dedent(f"""
                <div class="feature-panel" style="min-height:190px;">
                    <div class="mini-label">MATERIAL CLASS</div>

                    <h2>{name}</h2>

                    <strong>{full_name}</strong>

                    <p style="color:#6b7b70; font-size:.88rem;">
                        {description}
                    </p>
                </div>
            """))

    # 04. MODEL EVALUATION
    section("MODEL EVALUATION")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Held-out test accuracy",
            TEST_ACCURACY
        )

    with col2:
        st.metric(
            "Test images",
            TEST_IMAGE_COUNT
        )

    st.html("""
        <div class="soft-note">
            These evaluation results describe the MobileNetV2
            classifier on its held-out test dataset. They are not
            a guarantee of accuracy on every real-world image.
        </div>
    """)

    # 05. FUTURE DEVELOPMENT
    section("WHAT COMES NEXT?")

    first, second = st.columns(2, gap="large")

    with first:

        st.html("""
            <div class="panel" style="height:100%;">
                <div class="mini-label">FUTURE DEVELOPMENT</div>

                <h3>Smart sorting infrastructure</h3>

                <p style="color:#6b7b70;">
                    Exploring how AI classification could eventually
                    support intelligent waste sorting facilities.
                </p>
            </div>
        """)

    with second:

        st.html("""
            <div class="panel" style="height:100%;">
                <div class="mini-label">FUTURE DEVELOPMENT</div>

                <h3>Automated material handling</h3>

                <p style="color:#6b7b70;">
                    Future development could investigate integrating
                    AI predictions with conveyor systems and physical separation.
                </p>
            </div>
        """)

    st.html("""
        <div class="soft-note">
            <strong>Current prototype status:</strong>
            EcoCycle currently performs image-based material
            classification using MobileNetV2. Object detection,
            conveyor integration and physical automated sorting
            are not active in this version.
        </div>
    """)

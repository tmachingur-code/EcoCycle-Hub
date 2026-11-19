import streamlit as st
from config.settings import CLASSIFIER_PATH


@st.cache_resource(show_spinner="Loading EcoCycle MobileNetV2 model...")
def load_classifier():
    """Load and cache the trained MobileNetV2 classifier."""

    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(
            f"MobileNetV2 model not found: {CLASSIFIER_PATH}"
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(CLASSIFIER_PATH)

    return model

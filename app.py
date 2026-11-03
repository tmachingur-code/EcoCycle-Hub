"""
EcoCycle Energy Hub — Three-Class Material Classification Prototype

"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from supabase import create_client
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE AND MODEL SETTINGS
# ============================================================

st.set_page_config(
    page_title="EcoCycle Energy Hub | Smart Waste Intelligence",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_ecocycle_3class_model.keras"
SUPABASE_TABLE = "classification_scans"

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["PET", "HDPEM", "OTHER"]

CLASS_DESCRIPTIONS = {
    "PET": "Polyethylene Terephthalate",
    "HDPEM": "High-Density Polyethylene",
    "OTHER": "Other materials, including glass and aluminium",
}

CLASS_GUIDANCE = {
    "PET": (
        "PET identified. PET is commonly used in beverage bottles "
        "and can be considered for PET recovery."
    ),
    "HDPEM": (
        "HDPE identified. This plastic is commonly used in milk "
        "containers and household packaging."
    ),
    "OTHER": (
        "Other material identified. This category includes examples "
        "such as glass and aluminium."
    ),
}

CLASS_ICONS = {
    "PET": "PET",
    "HDPEM": "HD",
    "OTHER": "OTH",
}

# These figures are the held-out evaluation results shown in the
# current prototype. Update them if the evaluation is run again.
TEST_ACCURACY = "95.30%"
TEST_IMAGE_COUNT = "723"


# ============================================================
# COLOUR PALETTE
# ============================================================

FOREST = "#166534"
FOREST_DARK = "#14532D"
TEAL = "#0F766E"
TEAL_DARK = "#115E59"
GREEN = "#22C55E"
ORANGE = "#F59E0B"

INK = "#0F172A"
MUTED = "#64748B"
EDGE = "#E2E8F0"
BG = "#F8FAFC"


# ============================================================
# CUSTOM CSS — PRESERVES THE ORIGINAL DESIGN
# ============================================================

CSS = f"""
<style>
:root {{ color-scheme: light; }}

html, body, .stApp {{
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont,
    'Segoe UI', sans-serif;
    font-size: 16px;
}}

.stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {{
    background: {BG} !important;
    color: {INK} !important;
}}

.block-container {{
    padding: 1rem 1.4rem 3rem 1.4rem;
    max-width: 1180px;
}}

header, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDeployButton"], #MainMenu, footer {{
    visibility: hidden;
}}

.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
    color: {INK} !important;
    letter-spacing: -0.025em;
    font-weight: 750;
}}

.stApp .stMarkdown, .stApp .stMarkdown p,
.stApp label, [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] {{
    color: {INK};
}}

[data-testid="stFileUploader"] section {{
    border-radius: 14px;
    border: 2px dashed #9FC8C2;
    background: #fff;
}}

[data-testid="stFileUploader"] * {{
    color: {INK} !important;
}}

[data-testid="stImage"] img {{
    border-radius: 14px;
}}

.stButton > button,
[data-testid^="stBaseButton-primary"] {{
    background: {FOREST} !important;
    border: 1px solid {FOREST} !important;
    color: #fff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    padding: .55rem 1.2rem !important;
}}

.stButton > button:hover {{
    filter: brightness(.94);
    transform: translateY(-1px);
}}

.stButton > button p {{
    color: #fff !important;
}}

/* Top bar */
.topbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    background: #fff;
    border: 1px solid {EDGE};
    border-radius: 14px;
    padding: 10px 16px;
    margin-bottom: 8px;
}}

.brand {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    font-size: 1.1rem;
    color: {INK};
}}

.brand .logo {{
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: {TEAL};
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
}}

.brand small {{
    display: block;
    font-weight: 600;
    font-size: .74rem;
    color: {MUTED};
    margin-top: -2px;
}}

.chips {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}}

.chip {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: .82rem;
    font-weight: 600;
    background: #F0F9F7;
    border: 1px solid #BFE3DD;
    color: {TEAL_DARK};
}}

.chip.plain {{
    background: #F3F4F6;
    border-color: {EDGE};
    color: {MUTED};
}}

/* Navigation */
.st-key-appnav {{
    position: sticky;
    top: 0;
    z-index: 90;
    background: rgba(243,246,248,.96);
    padding: 2px 0 0 0;
    margin-bottom: 14px;
    border-bottom: 1px solid {EDGE};
}}

.st-key-appnav [data-testid="stButtonGroup"] {{
    gap: 2px;
    flex-wrap: wrap;
}}

.st-key-appnav button {{
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    color: {MUTED} !important;
    font-weight: 600 !important;
    padding: .6rem 1rem !important;
}}

.st-key-appnav button p {{
    color: inherit !important;
}}

.st-key-appnav button[aria-checked="true"] {{
    color: {TEAL} !important;
    border-bottom-color: {TEAL} !important;
    font-weight: 700 !important;
}}

/* Hero */
.hero {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    background: linear-gradient(135deg, {FOREST_DARK} 0%, #166534 52%, {TEAL} 100%);
    border-radius: 20px;
    padding: 34px 36px;
    margin: 4px 0 18px 0;
}}

.hero .copy {{
    flex: 1 1 320px;
    min-width: 0;
}}

.hero .hero-title {{
    color: #fff !important;
    font-size: 2.55rem;
    line-height: 1.12;
    font-weight: 800;
    margin: 0 0 12px 0;
}}

.hero p {{
    color: #D3EFEA !important;
    font-size: 1.08rem;
    margin: 0;
    max-width: 34em;
    line-height: 1.55;
}}

.hero-art {{
    font-size: 92px;
    padding: 8px 22px;
}}

.pagehead {{
    margin: 4px 0 14px 0;
}}

.pagehead .ph-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {INK};
}}

.pagehead p {{
    color: {MUTED} !important;
    margin: 2px 0 0 0;
    font-size: 1.02rem;
}}

.sec {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {INK};
    margin: 22px 0 8px 0;
}}

/* KPI and content cards */
.grid {{
    display: grid;
    gap: 14px;
    margin: 6px 0 12px 0;
}}

.grid.kpis {{
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}}

.grid.cards {{
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}}

.grid.cards.wide {{
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
}}

.kpi, .card {{
    transition: transform .18s ease, box-shadow .18s ease,
    border-color .18s ease;
}}

.kpi {{
    background: #fff;
    border: 1px solid {EDGE};
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(15,31,46,.04);
}}

.k-label {{
    color: {MUTED};
    font-size: .9rem;
    font-weight: 600;
}}

.k-value {{
    color: {INK};
    font-size: 2.1rem;
    font-weight: 800;
    line-height: 1.2;
    margin: 2px 0;
}}

.k-sub {{
    color: #52606D;
    font-size: .84rem;
}}

.kpi.green .k-value {{ color: {GREEN}; }}
.kpi.teal .k-value {{ color: {TEAL}; }}

.card {{
    background: #fff;
    border: 1px solid {EDGE};
    border-radius: 16px;
    padding: 19px 20px;
    box-shadow: 0 1px 2px rgba(15,31,46,.04);
}}

.card .ib {{
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: #E6F4F1;
    color: {TEAL};
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 12px;
    font-size: 24px;
}}

.card .c-title {{
    margin: 0 0 6px 0;
    font-size: 1.05rem;
    font-weight: 700;
    color: {INK};
}}

.card p {{
    margin: 0;
    color: #344054 !important;
    font-size: .95rem;
    line-height: 1.55;
}}

.card:hover, .kpi:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 26px rgba(15,31,46,.08);
    border-color: #8CCFC5;
}}

/* Numbered process cards on the How it works page */
.flow {{
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 14px;
    margin: 12px 0 22px 0;
}}

.flow .step {{
    position: relative;
    min-height: 164px;
    padding: 20px 15px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    text-align: center;
    background: #F0F9F7;
    border: 1px solid #8CCFC5;
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(15,31,46,.04);
    transition: transform .18s ease, box-shadow .18s ease;
}}

.flow .step:hover {{
    transform: translateY(-3px);
    box-shadow: 0 10px 22px rgba(15,31,46,.08);
}}

.flow .step .em {{
    width: 46px;
    height: 46px;
    margin: 0 auto 12px auto;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: {TEAL};
    color: #fff;
    font-size: 1.1rem;
    font-weight: 800;
}}

.flow .step b {{
    display: block;
    color: {INK};
    font-size: .98rem;
    line-height: 1.3;
    margin-bottom: 7px;
}}

.flow .step span {{
    display: block;
    color: {MUTED};
    font-size: .87rem;
    line-height: 1.45;
}}

.result-card {{
    background: #E7F5EB;
    border: 1px solid #B7DFC3;
    border-radius: 16px;
    padding: 22px;
    margin: 8px 0 14px 0;
}}

.result-title {{
    color: #14532D;
    font-size: 1.7rem;
    font-weight: 800;
}}

.result-description {{
    color: #42634D;
    font-size: .95rem;
}}

.decision {{
    border-radius: 14px;
    padding: 16px 20px;
    margin: 10px 0;
    font-weight: 700;
}}

.decision.idle {{
    background: #fff;
    border: 1px solid {EDGE};
    color: {MUTED};
}}

.note {{
    color: {MUTED} !important;
    font-size: .88rem;
}}

.appfoot {{
    text-align: center;
    color: #667085;
    font-size: .84rem;
    margin-top: 34px;
    padding-top: 14px;
    border-top: 1px solid {EDGE};
}}

@media (max-width: 640px) {{
    .block-container {{ padding: .6rem .8rem 2rem .8rem; }}
    .hero {{ padding: 24px 20px; border-radius: 16px; }}
    .hero .hero-title {{ font-size: 1.9rem; }}
    .hero p {{ font-size: 1rem; }}
    .hero-art {{ font-size: 60px; }}
    .pagehead .ph-title {{ font-size: 1.5rem; }}
    .kpi {{ padding: 12px 14px; }}
    .k-value {{ font-size: 1.7rem; }}
    .flow {{ grid-template-columns: 1fr; gap: 10px; }}
    .flow .step {{ min-height: auto; padding: 16px; }}
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HTML HELPERS — HTML IS BUILT AS SINGLE-LINE STRINGS
# ============================================================

def card(title, body, emoji="ECO"):
    return (
        f'<div class="card"><div class="ib">{emoji}</div>'
        f'<div class="c-title">{title}</div><p>{body}</p></div>'
    )


def card_grid(cards, wide=False):
    class_name = "grid cards wide" if wide else "grid cards"
    return f'<div class="{class_name}">{"".join(cards)}</div>'


def kpi(label, value, sub="", tone=""):
    return (
        f'<div class="kpi {tone}"><div class="k-label">{label}</div>'
        f'<div class="k-value">{value}</div>'
        f'<div class="k-sub">{sub}</div></div>'
    )


def kpi_grid(items):
    return '<div class="grid kpis">' + "".join(kpi(*item) for item in items) + "</div>"


def head(title, subtitle=""):
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="pagehead"><div class="ph-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def sec(title):
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


# ============================================================
# MODEL LOADING AND IMAGE PREDICTION
# ============================================================

@st.cache_resource(show_spinner="Loading the EcoCycle AI model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}. "
            "Place the trained three-class model beside app.py."
        )

    loaded_model = tf.keras.models.load_model(MODEL_PATH)

    # Warm up once to reduce the delay on the first prediction.
    dummy = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    loaded_model(dummy, training=False)

    return loaded_model


def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image, loaded_model):
    processed = prepare_image(image)

    # Direct model call avoids the extra overhead of model.predict().
    output = loaded_model(
        tf.convert_to_tensor(processed),
        training=False,
    ).numpy()[0]

    if len(output) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, "
            f"but received {len(output)}. Check the model/class order."
        )

    predicted_index = int(np.argmax(output))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(output[predicted_index]) * 100

    probabilities = {
        CLASS_NAMES[index]: float(output[index]) * 100
        for index in range(len(CLASS_NAMES))
    }

    return predicted_class, confidence, probabilities


# ============================================================
# SUPABASE DATABASE — SHARED CLASSIFICATION HISTORY
# ============================================================

HISTORY_COLUMNS = [
    "timestamp",
    "input_method",
    "predicted_class",
    "confidence",
    "pet_probability",
    "hdpem_probability",
    "other_probability",
]


@st.cache_resource
def get_supabase_client():
    """Create one reusable Supabase client from Streamlit secrets."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def load_history():
    """Read all scan rows from Supabase in pages."""
    client = get_supabase_client()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to .streamlit/secrets.toml before using scan history."
        )

    rows = []
    page_size = 1000
    start = 0

    while True:
        response = (
            client.table(SUPABASE_TABLE)
            .select("id, created_at, input_method, predicted_class, confidence, "
                    "pet_probability, hdpem_probability, other_probability")
            .order("created_at", desc=True)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size

    if not rows:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    history = pd.DataFrame(rows).rename(columns={"created_at": "timestamp"})
    return history[HISTORY_COLUMNS]


def save_prediction(source_name, predicted_class, confidence, probabilities):
    """Save a classification result to the shared Supabase table."""
    client = get_supabase_client()
    if client is None:
        raise RuntimeError(
            "Supabase is not configured. Add SUPABASE_URL and SUPABASE_KEY "
            "to .streamlit/secrets.toml."
        )

    # Store only whether the image came from the camera or an upload.
    # The original image and local filename are not uploaded or retained.
    input_method = "camera" if source_name == "camera_capture.jpg" else "upload"

    record = {
        "input_method": input_method,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 2),
        "pet_probability": round(float(probabilities["PET"]), 2),
        "hdpem_probability": round(float(probabilities["HDPEM"]), 2),
        "other_probability": round(float(probabilities["OTHER"]), 2),
    }

    client.table(SUPABASE_TABLE).insert(record).execute()


# ============================================================
# TOP BAR AND NAVIGATION
# ============================================================

st.markdown(
    '<div class="topbar">'
    '<div class="brand"><div class="logo">E</div>'
    '<div>EcoCycle Energy Hub<small>Smart Waste Intelligence</small></div></div>'
    '<div class="chips"><span class="chip">AI material identification</span>'
    '<span class="chip plain">Three-class prototype</span></div></div>',
    unsafe_allow_html=True,
)

PAGES = ["Home", "AI Scanner", "Dashboard", "How it works", "About"]

if "nav" not in st.session_state or st.session_state.nav not in PAGES:
    st.session_state.nav = "Home"

with st.container(key="appnav"):
    try:
        page = st.pills(
            "Menu",
            PAGES,
            selection_mode="single",
            key="nav",
            label_visibility="collapsed",
        )
    except AttributeError:
        page = st.radio(
            "Menu",
            PAGES,
            key="nav",
            horizontal=True,
            label_visibility="collapsed",
        )

page = page or "Home"


# ============================================================
# LOAD HISTORY AND MODEL
# ============================================================

history_error = None
try:
    history = load_history()
except Exception as error:
    history = pd.DataFrame(columns=HISTORY_COLUMNS)
    history_error = str(error)

try:
    model = load_model()
    model_error = None
except Exception as error:
    model = None
    model_error = str(error)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.markdown(
        '<div class="hero"><div class="copy">'
        '<div class="hero-title">Smarter Waste.<br>Cleaner Future.</div>'
        '<p>Using artificial intelligence and computer vision to identify '
        'recyclable materials and support smarter waste recovery.</p>'
        '</div><div class="hero-art">PET<br>01</div></div>',
        unsafe_allow_html=True,
    )

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button(
        "Open AI Scanner",
        type="primary",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"nav": "AI Scanner"}),
    )
    b2.button(
        "How it works",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"nav": "How it works"}),
    )

    sec("Our Digital Prototype")
    st.markdown(
        card_grid([
            card(
                "PET Classification",
                "Identify PET packaging, including commonly used beverage bottles.",
                "PET",
            ),
            card(
                "HDPEM Classification",
                "Recognise high-density polyethylene packaging such as milk containers.",
                "HD",
            ),
            card(
                "Other Materials",
                "Identify examples such as glass and aluminium as part of the OTHER category.",
                "OTH",
            ),
        ]),
        unsafe_allow_html=True,
    )

    sec("Model Performance")
    st.markdown(
        kpi_grid([
            ("Test accuracy", TEST_ACCURACY, "Held-out evaluation dataset", "green"),
            ("Classification classes", "3", "PET, HDPEM and OTHER", "teal"),
            ("Test images", TEST_IMAGE_COUNT, "Images in the held-out test set", ""),
            ("Images analysed", len(history), "Shared database records", ""),
        ]),
        unsafe_allow_html=True,
    )

    st.caption(
        "Evaluation figures are from the current model test. "
        "Predictions on new images may differ."
    )

    sec("Why EcoCycle Hub?")
    st.markdown(
        card_grid([
            card(
                "AI-assisted identification",
                "Uses computer vision to estimate the material category from an uploaded image.",
                "🤖",
            ),
            card(
                "Material awareness",
                "Helps distinguish PET and HDPEM plastics from the OTHER category.",
                "🔎",
            ),
            card(
                "Circular economy focus",
                "Explores how better material identification can support recyclable material recovery.",
                "🌱",
            ),
        ]),
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(f"AI model unavailable: {model_error}")
    else:
        st.info(
            "EcoCycle Hub currently identifies materials from images. "
            "It does not physically sort or separate them."
        )


# ============================================================
# AI SCANNER PAGE
# ============================================================

elif page == "AI Scanner":

    head(
        "AI Material Scanner",
        "Take a photo of a material or choose an image from your device.",
    )

    if model_error:
        st.error(f"Unable to load the AI model: {model_error}")
        st.stop()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        sec("Capture material")
        camera_file = st.camera_input(
            "Point your camera at the material and take a photo",
            key="material_camera",
        )

        st.caption("Prefer to use an existing image?")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            key="material_upload",
        )

        image_source = camera_file if camera_file is not None else uploaded_file
        source_name = (
            "camera_capture.jpg"
            if camera_file is not None
            else (uploaded_file.name if uploaded_file is not None else "")
        )

        image = None
        image_bytes = None
        if image_source is not None:
            try:
                image_bytes = image_source.getvalue()
                image = Image.open(image_source).convert("RGB")
                st.image(
                    image,
                    caption="Image ready for identification",
                    use_container_width=True,
                )
            except Exception:
                st.error("Unable to read this image. Please take another photo or choose a different file.")

    with right:
        sec("Identification result")

        if image is None:
            st.markdown(
                '<div class="card"><div class="ib">READY</div>'
                '<div class="c-title">Ready to identify a material</div>'
                '<p>Take a photo with your camera. The model will analyse it automatically.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            import hashlib
            image_key = hashlib.sha256(image_bytes).hexdigest()
            last_key = st.session_state.get("last_analysed_image")

            if image_key != last_key:
                try:
                    with st.spinner("Identifying material..."):
                        predicted_class, confidence, probabilities = predict_image(
                            image, model
                        )

                    st.session_state["latest_prediction"] = {
                        "class": predicted_class,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "source": source_name,
                    }
                    st.session_state["last_analysed_image"] = image_key

                    try:
                        save_prediction(
                            source_name,
                            predicted_class,
                            confidence,
                            probabilities,
                        )
                        st.session_state["latest_save_status"] = "saved"
                    except Exception as database_error:
                        st.session_state["latest_save_status"] = str(database_error)
                except Exception as error:
                    st.error(f"Could not analyse this image: {error}")

            result = st.session_state.get("latest_prediction")
            if result:
                predicted_class = result["class"]
                confidence = result["confidence"]
                probabilities = result["probabilities"]

                st.markdown(
                    f'<div class="result-card">'
                    f'<div style="font-size:13px;font-weight:800;letter-spacing:.05em;">MATERIAL IDENTIFIED</div>'
                    f'<div class="result-title">{predicted_class}</div>'
                    f'<div class="result-description">{CLASS_DESCRIPTIONS[predicted_class]}</div></div>',
                    unsafe_allow_html=True,
                )

                st.metric("Prediction confidence", f"{confidence:.2f}%")
                st.progress(min(max(confidence / 100, 0.0), 1.0))

                sec("Class probabilities")
                for category, score in probabilities.items():
                    st.write(f"**{category}** — {score:.2f}%")
                    st.progress(min(max(score / 100, 0.0), 1.0))

                st.success(CLASS_GUIDANCE[predicted_class])
                save_status = st.session_state.get("latest_save_status")
                if save_status == "saved":
                    st.caption("Scan result saved to the shared EcoCycle database.")
                elif save_status:
                    st.warning(f"This result was analysed but not saved to the shared database: {save_status}")
                st.caption(
                    "The confidence score is the model's estimated probability, "
                    "not a guarantee. The model selects one of its three trained classes."
                )

# ============================================================
# DASHBOARD PAGE
# ============================================================

elif page == "Dashboard":

    head(
        "Classification Dashboard",
        "Shared scan records submitted by users of EcoCycle Hub.",
    )

    refresh_col, _ = st.columns([1, 5])
    refresh_col.button("Refresh dashboard", on_click=st.rerun, use_container_width=True)

    if history_error:
        st.error(f"Unable to load shared scan history: {history_error}")
        st.info("Check your Supabase connection and database table setup.")
    elif history.empty:
        st.markdown(
            '<div class="decision idle">No scans recorded yet. '
            'Open the AI Scanner and analyse an image to begin.</div>',
            unsafe_allow_html=True,
        )
    else:
        total_scans = len(history)
        pet_count = int((history["predicted_class"] == "PET").sum())
        hdpem_count = int((history["predicted_class"] == "HDPEM").sum())
        other_count = int((history["predicted_class"] == "OTHER").sum())

        st.markdown(
            kpi_grid([
                ("Images analysed", total_scans, "All saved image scans", "green"),
                ("PET predictions", pet_count, "Predicted as PET", "teal"),
                ("HDPEM predictions", hdpem_count, "Predicted as HDPEM", ""),
                ("OTHER predictions", other_count, "Predicted as OTHER", ""),
            ]),
            unsafe_allow_html=True,
        )

        sec("Classification breakdown")

        breakdown = pd.DataFrame({
            "Material": ["PET", "HDPEM", "OTHER"],
            "Predictions": [pet_count, hdpem_count, other_count],
        })

        st.bar_chart(breakdown.set_index("Material"))

        sec("Recent scans")
        display_history = history.copy()
        display_history = display_history.sort_values(
            "timestamp", ascending=False
        )
        st.dataframe(display_history, hide_index=True, use_container_width=True)

        st.caption("Only classification results are stored. Original images and personal filenames are not saved.")

        st.download_button(
            "Download scan history (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="ecocycle_classification_history.csv",
            mime="text/csv",
        )


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif page == "How it works":

    head("How it works", "Three simple steps to identify a material.")

    steps = [
        ("01", "Take a photo", "Capture the material with your phone camera."),
        ("02", "AI identifies it", "The model classifies the image as PET, HDPEM or OTHER."),
        ("03", "View your result", "See the predicted material and its confidence score."),
    ]

    st.markdown(
        '<div class="flow">'
        + "".join(
            f'<div class="step"><div class="em">{number}</div>'
            f'<b>{title}</b><span>{description}</span></div>'
            for number, title, description in steps
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    sec("Technology")
    st.markdown(
        card_grid([
            card("Python", "Application and image processing.", "PY"),
            card("Streamlit", "Camera capture and user interface.", "WEB"),
            card("TensorFlow / Keras", "Runs the trained classification model.", "ML"),
            card("MobileNetV2", "Image classification architecture.", "CV"),
        ]),
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT PAGE — ORIGINAL WHO WE ARE / MISSION / VISION / VALUES LAYOUT
# ============================================================

elif page == "About":

    head("About us")

    sec("Who we are")
    st.markdown(
        card_grid([
            card(
                "EcoCycle Energy Hub",
                "A circular-economy initiative exploring how accessible technology can help identify recyclable materials and support resource recovery in communities.",
                "ECO",
            ),
        ], wide=True),
        unsafe_allow_html=True,
    )

    st.markdown(
        card_grid([
            card(
                "Our mission",
                "Turn urban waste into sustainable value by exploring practical technology for material identification and recovery.",
                "MIS",
            ),
            card(
                "Our vision",
                "A cleaner, more resilient Africa where waste is recognised as a resource, not simply a problem.",
                "VIS",
            ),
        ], wide=True),
        unsafe_allow_html=True,
    )

    sec("Our core values")
    st.markdown(
        card_grid([
            card(
                "Community first",
                "Technology should support people and the communities where it is used.",
                "COM",
            ),
            card(
                "Circularity",
                "Recover useful materials and keep resources in circulation.",
                "CIR",
            ),
            card(
                "Accessible technology",
                "Build practical tools that are clear and straightforward to use.",
                "ACC",
            ),
            card(
                "Responsible data",
                "Use classification information responsibly and avoid collecting unnecessary personal data.",
                "DATA",
            ),
        ]),
        unsafe_allow_html=True,
    )

    sec("Our current prototype")
    st.markdown(
        card_grid([
            card("PET", CLASS_DESCRIPTIONS["PET"], "PET"),
            card("HDPEM", CLASS_DESCRIPTIONS["HDPEM"], "HD"),
            card("OTHER", CLASS_DESCRIPTIONS["OTHER"], "OTH"),
        ]),
        unsafe_allow_html=True,
    )

    st.info(
        "The current application identifies materials from a camera photo or uploaded image. "
        "It does not physically sort or separate materials."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="appfoot"><strong style="color:#166534;">EcoCycle Energy Hub</strong>'
    '<br>AI-powered material identification for a circular economy.'
    '</div>',
    unsafe_allow_html=True,
)

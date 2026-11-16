"""
EcoCycle Energy Hub — Three-Class Material Classification Prototype

Run:
    streamlit run EcoCycle_Energy_Hub_app.py

Keep best_ecocycle_3class_model.keras in the same folder as this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# PAGE AND MODEL SETTINGS
# ============================================================

st.set_page_config(
    page_title="EcoCycle Energy Hub | Smart Material Intelligence",
    page_icon="E",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_ecocycle_3class_model.keras"
ASSET_DIR = BASE_DIR / "assets"

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
BG = "#F3F6EF"


# ============================================================
# CUSTOM CSS — PRESERVES THE ORIGINAL DESIGN
# ============================================================

CSS = f"""
<style>
:root {{ color-scheme: light; }}
html, body, .stApp {{ font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:{INK}; }}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {{
  background: #F6F8F5 !important; color:{INK} !important;
}}
.block-container {{ max-width: 1240px; padding: 1.25rem clamp(.8rem, 3vw, 2.2rem) 3rem; }}
header, [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDeployButton"], #MainMenu, footer {{ visibility:hidden; }}
.stApp h1,.stApp h2,.stApp h3,.stApp h4 {{ color:{INK} !important; letter-spacing:-.035em; font-weight:750; }}
.stApp p,.stApp label,[data-testid="stWidgetLabel"] p {{ color:{INK}; }}
[data-testid="stCaptionContainer"] p,.note {{ color:{MUTED} !important; }}
[data-testid="stFileUploader"] section {{ border:1.5px dashed #A9C7B9; border-radius:15px; background:#FBFDFC; }}
[data-testid="stFileUploader"] * {{ color:{INK} !important; }}
[data-testid="stImage"] img {{ border-radius:16px; }}
.stButton > button,[data-testid^="stBaseButton-primary"] {{ background:{FOREST} !important; border:1px solid {FOREST} !important; color:#fff !important; font-weight:650 !important; border-radius:11px !important; min-height:2.65rem; padding:.55rem 1.1rem !important; transition:all .18s ease; }}
.stButton > button:hover {{ background:{TEAL_DARK} !important; border-color:{TEAL_DARK} !important; transform:translateY(-1px); }}
.stButton > button p {{ color:#fff !important; }}
/* Brand header */
.topbar {{ display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; background:#fff; border:1px solid #E4EBE5; border-radius:17px; padding:13px 18px; margin:0 0 13px; box-shadow:0 4px 18px rgba(23,54,39,.035); }}
.brand {{ display:flex; align-items:center; gap:12px; color:{INK}; font-weight:800; font-size:1.05rem; letter-spacing:-.02em; }}
.brand .logo {{ width:39px; height:39px; border-radius:12px; background:{FOREST}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:1rem; font-weight:800; letter-spacing:-.04em; }}
.brand small {{ display:block; color:{MUTED}; font-size:.73rem; font-weight:550; margin-top:1px; letter-spacing:0; }}
.chips {{ display:flex; flex-wrap:wrap; gap:7px; }}
.chip {{ display:inline-flex; border-radius:999px; padding:6px 11px; font-size:.76rem; font-weight:650; background:#EAF4EE; border:1px solid #D7E8DC; color:{FOREST_DARK}; }}
.chip.plain {{ background:#F7F9F7; border-color:{EDGE}; color:{MUTED}; }}
/* Sticky horizontal navigation */
.st-key-appnav {{ position:sticky; top:0; z-index:90; background:rgba(246,248,245,.96); backdrop-filter:blur(12px); padding:4px 0 8px; margin-bottom:18px; border-bottom:1px solid #E3EAE4; }}
.st-key-appnav [data-testid="stButtonGroup"] {{ gap:4px; flex-wrap:wrap; }}
.st-key-appnav button {{ background:transparent !important; border:1px solid transparent !important; border-radius:9px !important; color:{MUTED} !important; font-weight:650 !important; padding:.48rem .78rem !important; min-height:2.35rem; }}
.st-key-appnav button p {{ color:inherit !important; font-size:.88rem; }}
.st-key-appnav button[aria-checked="true"] {{ background:#E6F1E9 !important; border-color:#D1E4D6 !important; color:{FOREST_DARK} !important; }}
/* Home hero */
.hero {{ position:relative; overflow:hidden; display:flex; align-items:center; justify-content:space-between; gap:24px; background:linear-gradient(112deg,#123B2D 0%,#185B43 58%,#16766C 100%); border-radius:23px; padding:clamp(27px,5vw,54px); margin:4px 0 18px; box-shadow:0 18px 42px rgba(18,59,45,.13); }}
.hero:after {{ content:""; position:absolute; width:330px; height:330px; border:1px solid rgba(255,255,255,.13); border-radius:50%; right:7%; top:-52%; box-shadow:0 0 0 34px rgba(255,255,255,.035),0 0 0 72px rgba(255,255,255,.025); }}
.hero .copy {{ position:relative; z-index:1; flex:1 1 350px; min-width:0; }}
.hero .eyebrow {{ color:#B9E5D0; font-size:.75rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; margin-bottom:13px; }}
.hero .hero-title {{ color:#fff !important; font-size:clamp(2.1rem,5vw,3.55rem); line-height:1.04; font-weight:800; letter-spacing:-.055em; margin:0 0 15px; }}
.hero p {{ color:#DCEFE5 !important; font-size:clamp(.98rem,1.6vw,1.1rem); max-width:39rem; line-height:1.65; margin:0; }}
.hero-art {{ position:relative; z-index:1; flex:0 0 210px; min-height:155px; display:flex; align-items:center; justify-content:center; }}
.hero-art .orbit {{ width:150px; height:150px; border:1px solid rgba(255,255,255,.35); border-radius:50%; display:flex; align-items:center; justify-content:center; box-shadow:0 0 0 17px rgba(255,255,255,.06),0 0 0 36px rgba(255,255,255,.035); }}
.hero-art .orbit span {{ display:flex; align-items:center; justify-content:center; width:82px; height:82px; border-radius:24px; background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.32); color:#fff; font-size:1.2rem; font-weight:800; letter-spacing:-.05em; }}
.hero-photo {{ width:min(34%,360px); min-width:220px; height:225px; object-fit:cover; border-radius:18px; border:1px solid rgba(255,255,255,.24); box-shadow:0 12px 28px rgba(0,0,0,.18); position:relative; z-index:1; }}
.home-feature {{ display:grid; grid-template-columns:1.1fr 2fr; gap:18px; align-items:stretch; margin:8px 0 20px; }}
.home-feature-lead {{ background:#EAF3EC; border:1px solid #D7E7DA; border-radius:16px; padding:20px; display:flex; flex-direction:column; justify-content:center; }}
.home-feature-lead b {{ color:{FOREST_DARK}; font-size:1.05rem; margin-bottom:6px; }}
.home-feature-lead p {{ color:#526158; margin:0; font-size:.9rem; line-height:1.5; }}
.home-class-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
.home-class {{ background:#fff; border:1px solid #E2EAE3; border-radius:15px; padding:17px; }}
.home-class strong {{ display:block; color:{FOREST_DARK}; font-size:1rem; margin-bottom:6px; }}
.home-class span {{ color:{MUTED}; font-size:.84rem; line-height:1.45; }}
.pagehead {{ margin:5px 0 18px; }}
.pagehead .ph-title {{ font-size:clamp(1.55rem,3vw,2rem); font-weight:800; letter-spacing:-.04em; color:{INK}; }}
.pagehead p {{ color:{MUTED} !important; margin:4px 0 0; font-size:.98rem; }}
.sec {{ font-size:1.08rem; font-weight:750; color:{INK}; margin:23px 0 10px; letter-spacing:-.02em; }}
.grid {{ display:grid; gap:13px; margin:7px 0 13px; }}
.grid.kpis {{ grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); }}
.grid.cards {{ grid-template-columns:repeat(auto-fit,minmax(min(100%,220px),1fr)); }}
.grid.cards.wide {{ grid-template-columns:repeat(auto-fit,minmax(min(100%,285px),1fr)); }}
.kpi,.card {{ transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease; }}
.kpi {{ background:#fff; border:1px solid #E5ECE6; border-radius:15px; padding:17px 18px; box-shadow:0 3px 12px rgba(20,50,35,.025); }}
.k-label {{ color:{MUTED}; font-size:.83rem; font-weight:650; }}
.k-value {{ color:{INK}; font-size:clamp(1.65rem,3vw,2.05rem); font-weight:800; line-height:1.25; margin:4px 0; letter-spacing:-.04em; }}
.k-sub {{ color:{MUTED}; font-size:.8rem; line-height:1.4; }}
.kpi.green .k-value {{ color:{FOREST}; }} .kpi.teal .k-value {{ color:{TEAL}; }}
.card {{ background:#fff; border:1px solid #E5ECE6; border-radius:16px; padding:19px 20px; box-shadow:0 3px 12px rgba(20,50,35,.025); }}
.card .ib {{ width:42px; height:34px; border-radius:9px; background:#EAF3EC; color:{FOREST}; display:flex; align-items:center; justify-content:center; margin-bottom:14px; font-size:.72rem; font-weight:800; letter-spacing:.04em; }}
.card .c-title {{ margin:0 0 7px; font-size:1rem; font-weight:750; color:{INK}; }}
.card p {{ margin:0; color:#526158 !important; font-size:.91rem; line-height:1.58; }}
.card:hover,.kpi:hover {{ transform:translateY(-2px); box-shadow:0 10px 24px rgba(20,50,35,.07); border-color:#C5D9CA; }}
.system-intro {{ background:linear-gradient(115deg,#153F30,#17675D); color:#F4FFF8; border-radius:18px; padding:24px 27px; margin:8px 0 18px; }}
.system-intro h3 {{ color:#fff !important; margin:0 0 8px; }} .system-intro p {{ color:#D8EDE3 !important; margin:0; line-height:1.65; }}
.system-stage {{ background:#fff; border:1px solid #E2EAE3; border-radius:15px; padding:16px; min-height:138px; box-shadow:0 3px 12px rgba(20,50,35,.025); }}
.system-stage .stage-no {{ color:{TEAL}; font-size:.72rem; font-weight:800; letter-spacing:.1em; margin-bottom:8px; }}
.system-stage b {{ color:{INK}; display:block; margin-bottom:6px; }} .system-stage p {{ color:#526158; font-size:.87rem; line-height:1.5; margin:0; }}
.reference-label {{ display:inline-block; background:#EAF3EC; color:{FOREST_DARK}; border:1px solid #D4E5D8; border-radius:999px; padding:4px 9px; font-size:.68rem; font-weight:800; letter-spacing:.04em; margin:5px 0 8px; }}
.machine-note {{ background:#FBF7EA; border:1px solid #EDE2BF; color:#665321; border-radius:12px; padding:13px 16px; margin:12px 0 18px; font-size:.88rem; line-height:1.55; }}
.flow {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:12px; margin:12px 0 22px; }}
.flow .step {{ min-height:150px; padding:19px 15px; display:flex; flex-direction:column; align-items:flex-start; background:#fff; border:1px solid #E2EAE3; border-radius:15px; }}
.flow .step .em {{ width:36px; height:36px; margin:0 0 13px; display:flex; align-items:center; justify-content:center; border-radius:10px; background:{FOREST}; color:#fff; font-size:.8rem; font-weight:800; }}
.flow .step b {{ color:{INK}; font-size:.95rem; line-height:1.3; margin-bottom:7px; }} .flow .step span {{ color:{MUTED}; font-size:.84rem; line-height:1.5; }}
.result-card {{ background:#EAF4EC; border:1px solid #D1E5D5; border-radius:16px; padding:20px; margin:8px 0 14px; }}
.result-title {{ color:{FOREST_DARK}; font-size:1.75rem; font-weight:850; letter-spacing:-.04em; }} .result-description {{ color:#526B59; font-size:.91rem; }}
.decision {{ border-radius:13px; padding:16px 18px; margin:10px 0; font-weight:650; }} .decision.idle {{ background:#fff; border:1px solid {EDGE}; color:{MUTED}; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid #E5ECE6; border-radius:13px; padding:12px 14px; }}
[data-testid="stProgressBar"] > div {{ background:#E4ECE5; }}
.appfoot {{ text-align:center; color:{MUTED}; font-size:.8rem; margin-top:36px; padding-top:17px; border-top:1px solid #E3EAE4; }}
@media (max-width:760px) {{
 .block-container {{ padding:.65rem .8rem 2rem; }}
 .topbar {{ align-items:flex-start; }} .chips {{ width:100%; }}
 .st-key-appnav {{ position:relative; }} .st-key-appnav button {{ padding:.42rem .58rem !important; }}
 .hero {{ border-radius:17px; padding:27px 22px; }} .hero-art {{ display:none; }} .hero-photo {{ width:100%; min-width:0; height:180px; }}
 .home-feature {{ grid-template-columns:1fr; }} .home-class-grid {{ grid-template-columns:1fr; }}
 .st-key-appnav [data-testid="stButtonGroup"] {{ flex-wrap:nowrap; overflow-x:auto; max-width:100%; padding-bottom:5px; }}
 .st-key-appnav button {{ white-space:nowrap; flex:0 0 auto; }}
 .grid.kpis {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
 .flow {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
}}
@media (max-width:430px) {{ .grid.kpis {{ grid-template-columns:1fr 1fr; }} .flow {{ grid-template-columns:1fr; }} .card {{ padding:16px; }} }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# HTML HELPERS — HTML IS BUILT AS SINGLE-LINE STRINGS
# ============================================================

def card(title, body, mark="ECO"):
    return (
        f'<div class="card"><div class="ib">{mark}</div>'
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
# SESSION-BASED CLASSIFICATION HISTORY
# ============================================================

HISTORY_COLUMNS = [
    "timestamp", "input_method", "predicted_class", "confidence",
    "pet_probability", "hdpem_probability", "other_probability",
]


def initialise_history():
    if "scan_history" not in st.session_state:
        st.session_state.scan_history = []


def get_history():
    initialise_history()
    return pd.DataFrame(st.session_state.scan_history, columns=HISTORY_COLUMNS)


def save_prediction(source_name, predicted_class, confidence, probabilities):
    initialise_history()
    input_method = "camera" if source_name == "camera_capture.jpg" else "upload"
    st.session_state.scan_history.append({
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_method": input_method,
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 2),
        "pet_probability": round(float(probabilities["PET"]), 2),
        "hdpem_probability": round(float(probabilities["HDPEM"]), 2),
        "other_probability": round(float(probabilities["OTHER"]), 2),
    })


# ============================================================
# TOP BAR AND NAVIGATION
# ============================================================

st.markdown(
    '<div class="topbar">'
    '<div class="brand"><div class="logo">E</div>'
    '<div>EcoCycle Energy Hub<small>Smart Material Intelligence</small></div></div>'
    '<div class="chips"><span class="chip">AI material identification</span>'
    '<span class="chip plain">Three-class prototype</span></div></div>',
    unsafe_allow_html=True,
)

PAGES = ["Home", "AI Scanner", "Dashboard", "Impact", "How it works", "Smart Sorting System", "About"]

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
# SESSION HISTORY AND MODEL
# ============================================================

initialise_history()
history = get_history()

model = None
model_error = None

if page == "AI Scanner":
    try:
        model = load_model()
    except Exception as error:
        model_error = str(error)


# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    hero_image = ASSET_DIR / "hero_recycling.jpg"
    st.markdown(
        '<div class="hero"><div class="copy">'
        '<div class="eyebrow">AI-powered material identification</div>'
        '<div class="hero-title">Smarter identification.<br>Better recovery.</div>'
        '<p>Identify recyclable materials from a single image.</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    if hero_image.exists():
        st.image(str(hero_image), caption="Recycling and material recovery", use_container_width=True)

    b1, b2, b3 = st.columns([1, 1, 1.45])
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
    b3.button(
        "Future sorting concept",
        use_container_width=True,
        on_click=lambda: st.session_state.update({"nav": "Smart Sorting System"}),
    )

    sec("What the AI identifies")
    st.markdown(
        '<div class="home-feature">'
        '<div class="home-feature-lead"><b>Three material categories</b>'
        '<p>Upload or capture an image to receive a predicted material and confidence score.</p></div>'
        '<div class="home-class-grid">'
        '<div class="home-class"><strong>PET</strong><span>Common beverage bottles</span></div>'
        '<div class="home-class"><strong>HDPEM</strong><span>HDPE milk packaging</span></div>'
        '<div class="home-class"><strong>OTHER</strong><span>Other materials outside the two target classes</span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    sec("Model performance")
    st.markdown(
        kpi_grid([
            ("Test accuracy", TEST_ACCURACY, "Held-out test set", "green"),
            ("Test images", TEST_IMAGE_COUNT, "Evaluation images", "teal"),
            ("Scans this session", len(history), "Current session only", ""),
        ]),
        unsafe_allow_html=True,
    )
    st.caption("95.30% is the model's held-out test accuracy; performance on new real-world images may differ.")


# ============================================================
# AI SCANNER PAGE
# ============================================================

elif page == "AI Scanner":

    head(
        "AI Material Scanner",
        "Capture or upload a material image for AI classification.",
    )

    if model_error:
        st.error(f"Unable to load the AI model: {model_error}")
        st.stop()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        sec("Capture material")
        camera_file = st.camera_input(
            "Capture a material image",
            key="material_camera",
        )

        st.caption("Or upload an image")
        uploaded_file = st.file_uploader(
            "Upload image",
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
                    caption="Image ready for scanning",
                    use_container_width=True,
                )
            except Exception:
                st.error("Unable to read this image. Please take another photo or choose a different file.")

    with right:
        sec("Identification result")

        if image is None:
            st.markdown(
                '<div class="card"><div class="ib">READY</div>'
                '<div class="c-title">Ready to scan</div>'
                '<p>Capture or upload an image to see its predicted category.</p></div>',
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

                    save_prediction(
                        source_name,
                        predicted_class,
                        confidence,
                        probabilities,
                    )
                    st.session_state["latest_save_status"] = "saved"
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
                    st.caption("Scan saved for this session.")
                elif save_status:
                    st.warning(f"This result was analysed but not saved: {save_status}")
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
        "Your scan results from this session.",
    )

    refresh_col, _ = st.columns([1, 5])
    def refresh_dashboard():
        st.rerun()

    refresh_col.button("Refresh dashboard", on_click=refresh_dashboard, use_container_width=True)

    history = get_history()

    if history.empty:
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
                ("Images analysed", total_scans, "This session", "green"),
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

        st.caption("History is session-based. Download your CSV before ending the session.")

        st.download_button(
            "Download scan history (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="ecocycle_classification_history.csv",
            mime="text/csv",
        )


# ============================================================
# IMPACT PAGE
# ============================================================

elif page == "Impact":

    head("Our potential impact", "Better identification. Better recovery.")

    st.markdown(
        '<div class="system-intro">'
        '<h3>From identification to resource recovery</h3>'
        '<p>EcoCycle Energy Hub aims to help plastic recovery operators identify materials '
        'more consistently. Real-world impact will be measured through testing.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        card_grid([
            card("Support sorting", "Identify PET and HDPE milk packaging from images.", "SORT"),
            card("Build material awareness", "Make plastic categories easier to recognise.", "AI"),
            card("Enable recovery", "Support workflows that keep recyclable materials in circulation.", "CIRC"),
        ]),
        unsafe_allow_html=True,
    )

    sec("How we will measure impact")
    stages = [
        ("01 / ACCURACY", "Real-world images", "Test different lighting, angles and backgrounds."),
        ("02 / USABILITY", "Operator feedback", "Assess fit with existing sorting workflows."),
        ("03 / OUTCOMES", "Pilot evaluation", "Measure identification time and sorting consistency."),
    ]

    cols = st.columns(3)
    for col, (number, title, description) in zip(cols, stages):
        with col:
            st.markdown(
                f'<div class="system-stage"><div class="stage-no">{number}</div>'
                f'<b>{title}</b><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.caption("These are intended outcomes and future evaluation measures, not results already achieved.")


# ============================================================
# HOW IT WORKS PAGE
# ============================================================

elif page == "How it works":

    head("How it works", "Capture. Classify. Review.")

    steps = [
        ("01", "Take a photo", "Take or upload a photo."),
        ("02", "AI identifies it", "AI predicts PET, HDPEM or OTHER."),
        ("03", "View your result", "View the result and confidence score."),
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
# SMART SORTING SYSTEM — PROPOSED PHYSICAL INTEGRATION
# ============================================================

elif page == "Smart Sorting System":

    head("A future smart sorting line", "How today's AI model could connect to sorting equipment.")

    st.markdown(
        '<div class="system-intro">'
        '<h3>Camera → AI → Sorting mechanism</h3>'
        '<p>The current app identifies material from an image. In a future system, '
        'a camera and controller could use that prediction to guide items into separate collection streams.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    stages = [
        ("01 / FEED", "Conveyor", "Items move past an inspection point."),
        ("02 / CAPTURE", "Camera", "A camera captures each item."),
        ("03 / CLASSIFY", "AI model", "The model predicts PET, HDPEM or OTHER."),
        ("04 / SEPARATE", "Diverter", "A future mechanism directs the item to a bin."),
    ]
    stage_cols = st.columns(4)
    for col, (number, title, description) in zip(stage_cols, stages):
        with col:
            st.markdown(
                f'<div class="system-stage"><div class="stage-no">{number}</div>'
                f'<b>{title}</b><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="machine-note"><strong>Current status:</strong> '
        'The working prototype is image classification only. Conveyor integration, '
        'camera automation and physical separation have not yet been built.</div>',
        unsafe_allow_html=True,
    )

    reference_images = [
        ("recycling_conveyor.jpg", "Conveyor-based material handling"),
        ("sorting_facility.jpg", "Material recovery facility"),
    ]
    available_images = [(name, label) for name, label in reference_images if (ASSET_DIR / name).exists()]
    if available_images:
        sec("Industry reference")
        cols = st.columns(len(available_images), gap="large")
        for col, (name, label) in zip(cols, available_images):
            with col:
                st.image(str(ASSET_DIR / name), caption=label, use_container_width=True)


# ============================================================
# ABOUT PAGE — ORIGINAL WHO WE ARE / MISSION / VISION / VALUES LAYOUT
# ============================================================

elif page == "About":

    head("About EcoCycle", "Technology for better material recovery.")

    sec("Who we are")
    st.markdown(
        card_grid([
            card(
                "EcoCycle Energy Hub",
                "Building practical tools that help identify recyclable materials.",
                "ECO",
            ),
        ], wide=True),
        unsafe_allow_html=True,
    )

    st.markdown(
        card_grid([
            card(
                "Our mission",
                "Support smarter material identification and resource recovery.",
                "MIS",
            ),
            card(
                "Our vision",
                "A circular economy where waste is treated as a resource.",
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



# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="appfoot"><strong style="color:#166534;">EcoCycle Energy Hub</strong>'
    '<br>AI-powered plastic identification.'
    '</div>',
    unsafe_allow_html=True,
)

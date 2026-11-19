"""Streamlit app: upload an image -> Cat / Dog prediction.
Run:  streamlit run app/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.predict import load_animal_gate, load_model, predict_pil  # noqa: E402

MODEL_PATH = ROOT / "models" / "final_model.keras"
MAX_MB = 10

st.set_page_config(
    page_title="PawDetector | Cat & Dog Recognition",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17231f;
        --muted: #6b7973;
        --green: #2f725d;
        --green-dark: #205542;
        --mint: #e7f2eb;
        --cream: #f7f8f4;
        --line: #e4e9e3;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 88% 0%, rgba(216, 236, 220, .55), transparent 34rem),
            var(--cream);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .stApp, .stApp p, .stApp label, [data-testid="stSidebar"] * {
        color: var(--ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1.5rem; }
    .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
    [data-testid="stSidebar"] {
        background: #f0f4ee;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    [data-testid="stFileUploader"] section {
        background: #fbfcf9;
        border: 1.5px dashed #b7c9bc;
        border-radius: 18px;
        padding: 1.1rem;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--green); }
    [data-testid="stFileUploader"] button {
        border: 0;
        border-radius: 10px;
        background: var(--green);
        color: #fff;
    }
    [data-testid="stFileUploader"] button:hover {
        background: var(--green-dark);
        color: #fff;
        border: 0;
    }
    [data-testid="stMetric"] {
        background: #f7faf6;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 1rem 1.1rem;
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); font-weight: 600; }
    [data-testid="stMetricValue"] { color: var(--green-dark); font-family: 'Manrope', sans-serif; }
    [data-testid="stProgressBar"] > div > div { background: var(--green); }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,.55);
        border-radius: 26px;
        background: linear-gradient(118deg, #e5f1e8 0%, #f2f5eb 56%, #f8f4e9 100%);
        box-shadow: 0 16px 42px rgba(34, 73, 53, .07);
    }
    .hero:after {
        content: '🐾';
        position: absolute;
        right: 3rem;
        top: 50%;
        transform: translateY(-50%) rotate(-18deg);
        font-size: 6rem;
        opacity: .14;
    }
    .eyebrow {
        margin: 0 0 .45rem;
        color: var(--green);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0;
        font-size: clamp(2.15rem, 5vw, 3.2rem);
        letter-spacing: -.055em;
        line-height: 1.12;
    }
    .hero p:last-child {
        max-width: 600px;
        margin: .7rem 0 0;
        color: #53665b;
        font-size: 1.02rem;
    }
    .section-label {
        margin: .2rem 0 .75rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.08rem;
        font-weight: 700;
    }
    .soft-note { color: var(--muted); font-size: .9rem; }
    .result-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: #fff;
        box-shadow: 0 8px 26px rgba(31, 57, 43, .055);
    }
    .result-kicker {
        margin: 0 0 .35rem;
        color: var(--muted);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .result-value {
        margin: 0;
        color: var(--green-dark);
        font-family: 'Manrope', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line);
        border-radius: 20px;
        background: rgba(255,255,255,.72);
    }
    @media (max-width: 700px) {
        .hero { padding: 1.6rem; }
        .hero:after { right: 1rem; font-size: 4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading classifiers...")
def get_models():
    return load_model(str(MODEL_PATH)), load_animal_gate()


st.markdown(
    """
    <div class="hero">
        <p class="eyebrow">AI-powered pet recognition</p>
        <h1>PawDetector</h1>
        <p>Meet the moment: upload a photo and discover whether your furry friend is a cat or a dog.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🐾 PawDetector")
    st.caption("A little AI for your four-legged friends.")
    st.divider()
    st.markdown("### Prediction settings")
    min_conf = st.slider(
        "Confidence threshold",
        0.50,
        0.99,
        0.70,
        0.01,
        help="Below this threshold, the result is marked as uncertain.",
    )
    st.markdown(
        '<p class="soft-note">Only cat and dog photos are supported.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Powered by a trained CNN and image recognition.")

if "history" not in st.session_state:
    st.session_state.history = []

if not MODEL_PATH.exists():
    st.error(f"Model not found at {MODEL_PATH}. Run the notebook first to train and save it.")
    st.stop()

st.markdown('<p class="section-label">Start with a photo</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="soft-note">Choose a clear image of one pet. JPG, PNG, WEBP, or BMP · up to 10 MB</p>',
    unsafe_allow_html=True,
)
file = st.file_uploader(
    "Drop an image here or browse your device",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if file is not None:
    try:
        if file.size > MAX_MB * 1024 * 1024:
            raise ValueError(f"File is larger than {MAX_MB} MB.")
        img = Image.open(file)
        img.load()
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read this file as an image ({e}). Please upload a valid image.")
        st.stop()

    t0 = time.time()
    model, animal_gate = get_models()
    res = predict_pil(model, img, animal_gate)
    ms = (time.time() - t0) * 1000

    image_col, result_col = st.columns([1.05, 0.95], gap="large")
    with image_col:
        with st.container(border=True):
            st.markdown('<p class="section-label">Your photo</p>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.caption(file.name)

    with result_col:
        with st.container(border=True):
            st.markdown('<p class="section-label">PawDetector says</p>', unsafe_allow_html=True)
            if not res["is_cat_or_dog"]:
                st.warning("Unable to predict — please upload a photo of a cat or dog.")
                shown = "Unable to predict"
            elif res["confidence"] < min_conf:
                st.warning(
                    f"Not quite sure — confidence is {res['confidence']:.1%}, "
                    f"below your {min_conf:.0%} threshold."
                )
                shown = "Uncertain"
            else:
                icon = "🐶" if res["label"] == "Dog" else "🐱"
                st.markdown(
                    f'<div class="result-card"><p class="result-kicker">Best match</p>'
                    f'<p class="result-value">{icon} {res["label"]}</p></div>',
                    unsafe_allow_html=True,
                )
                shown = res["label"]

            if res["is_cat_or_dog"]:
                st.write("")
                c1, c2 = st.columns(2)
                c1.metric("Dog", f"{res['prob_dog']:.1%}")
                c2.metric("Cat", f"{res['prob_cat']:.1%}")
                st.progress(res["confidence"], text="Model confidence")
            st.caption(f"Analysis time · {ms:.0f} ms")

    st.session_state.history.append({"file": file.name, "prediction": shown,
                                     "P(Dog)": (round(res["prob_dog"], 4)
                                                if res["prob_dog"] is not None else "N/A"),
                                     "confidence": (round(res["confidence"], 4)
                                                    if res["is_cat_or_dog"] else "N/A")})

if st.session_state.history:
    st.divider()
    st.markdown('<p class="section-label">Recent predictions</p>', unsafe_allow_html=True)
    st.dataframe(st.session_state.history, use_container_width=True, hide_index=True)

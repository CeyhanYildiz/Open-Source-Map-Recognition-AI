import streamlit as st
from PIL import Image
import io
import time
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipForImageClassification
import pyautogui
import numpy as np
import os
import random

# --------------------------------
# CSS FIXED (TITLE NO LONGER CUT)
# --------------------------------
st.markdown("""
<style>
.small-text { font-size: 0.85rem; margin-bottom: -3px; }

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0.5rem !important;
}

h1 {
    margin-top: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Config / constants
# -------------------------
BASE_MODEL = "google/siglip-base-patch16-224"
MODEL_A_PATH = "./checkpoints/checkpoint-28130"
MODEL_B_NAME = "prithivMLmods/GeoGuessr-55"

LOCAL_IMAGE_BASE = r"C:/Users/ceyha/OneDrive/Documenten/Image"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Mapping for model B labels
id2label_B = {
    "0": "Argentina","1": "Australia","2": "Austria","3": "Bangladesh","4": "Belgium",
    "5": "Bolivia","6": "Botswana","7": "Brazil","8": "Bulgaria","9": "Cambodia",
    "10": "Canada","11": "Chile","12": "Colombia","13": "Croatia","14": "Czechia",
    "15": "Denmark","16": "Finland","17": "France","18": "Germany","19": "Ghana",
    "20": "Greece","21": "Hungary","22": "India","23": "Indonesia","24": "Ireland",
    "25": "Israel","26": "Italy","27": "Japan","28": "Kenya","29": "Latvia",
    "30": "Lithuania","31": "Malaysia","32": "Mexico","33": "Netherlands",
    "34": "New Zealand","35": "Nigeria","36": "Norway","37": "Peru","38": "Philippines",
    "39": "Poland","40": "Portugal","41": "Romania","42": "Russia","43": "Singapore",
    "44": "Slovakia","45": "South Africa","46": "South Korea","47": "Spain","48": "Sweden",
    "49": "Switzerland","50": "Taiwan","51": "Thailand","52": "Turkey","53": "Ukraine",
    "54": "United Kingdom"
}

# -------------------------
# Extract country from file path
# -------------------------
def extract_country_from_path(path: str):
    try:
        return os.path.basename(os.path.dirname(path))
    except:
        return None

# -------------------------
# Load models once
# -------------------------
@st.cache_resource
def load_model_A():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(MODEL_A_PATH).to(device)
    return processor, model

@st.cache_resource
def load_model_B():
    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = SiglipForImageClassification.from_pretrained(MODEL_B_NAME).to(device)
    return processor, model

processor_A, model_A = load_model_A()
processor_B, model_B = load_model_B()

# -------------------------
# Random local image picker
# -------------------------
def get_random_local_image():
    try:
        folders = [f.path for f in os.scandir(LOCAL_IMAGE_BASE) if f.is_dir()]
        if not folders:
            return None, None

        chosen_folder = random.choice(folders)
        images = [
            f.path for f in os.scandir(chosen_folder)
            if f.is_file() and f.name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            return None, None

        chosen_image = random.choice(images)
        return Image.open(chosen_image).convert("RGB"), chosen_image

    except Exception as e:
        st.error(f"Error picking image: {e}")
        return None, None

# -------------------------
# Prediction functions
# -------------------------
def predict_A(img: Image.Image, topk=5):
    inputs = processor_A(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model_A(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    topk_idx = probs.argsort()[-topk:][::-1]
    return [(model_A.config.id2label[int(i)], float(probs[int(i)] * 100.0)) for i in topk_idx]

def predict_B(img: Image.Image, topk=5):
    inputs = processor_B(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model_B(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    topk_idx = probs.argsort()[-topk:][::-1]
    return [(id2label_B[str(int(i))], float(probs[int(i)] * 100.0)) for i in topk_idx]

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Dual Model Image Dashboard", layout="wide")

st.title("Dual-Model Image Classification Dashboard")

# Model toggles
use_model_A = st.sidebar.checkbox("Enable Model A (SIGLIP Checkpoint)", value=True)
use_model_B = st.sidebar.checkbox("Enable Model B (GeoGuessr-55)", value=True)

st.sidebar.markdown("### Model Paths")
st.sidebar.caption(f"Model A folder: **{MODEL_A_PATH}**")
st.sidebar.caption(f"Model B repo: **{MODEL_B_NAME}**")

mode = st.sidebar.selectbox(
    "Input mode",
    ["Upload image", "Live screenshot (local)", "Example image URL", "Random Local Image (auto)"]
)

if mode == "Live screenshot (local)":
    interval = st.sidebar.slider("Screenshot interval (s)", 0.5, 5.0, 1.0, 0.5)

crop_area = st.sidebar.checkbox("Crop center 60%", value=False)

col_image, col_models = st.columns([0.9, 1.1], gap="small")

with col_image:
    st.subheader("Input Image")
    img = None
    chosen_path = None

    if mode == "Upload image":
        uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    elif mode == "Example image URL":
        url = st.text_input("Enter image URL:")
        if url:
            try:
                import requests
                resp = requests.get(url, timeout=10)
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except:
                st.error("Failed to fetch URL.")

    elif mode == "Live screenshot (local)":
        start = st.button("Start")
        stop = st.button("Stop")

    elif mode == "Random Local Image (auto)":
        run = st.checkbox("Start random mode")

with col_models:
    st.subheader("Predictions")
    cols = st.columns([1, 1], gap="small")
    placeholder_A = cols[0]
    placeholder_B = cols[1]

img_display = st.empty()

# -------------------------
# Render predictions
# -------------------------
def render_predictions(img: Image.Image):
    display_img = img

    if crop_area:
        w, h = img.size
        cw, ch = int(w * 0.6), int(h * 0.6)
        left = (w - cw) // 2
        top = (h - ch) // 2
        display_img = img.crop((left, top, left + cw, top + ch))

    img_display.image(display_img, width=460)

    # Model A
    if use_model_A:
        predsA = predict_A(display_img)
        with placeholder_A:
            st.markdown("**Model A — SIGLIP Checkpoint**")
            for label, score in predsA:
                st.markdown(f"<div class='small-text'>{label}: {score:.1f}%</div>", unsafe_allow_html=True)
            st.bar_chart({'score': [s for _, s in predsA]}, height=120)
    else:
        with placeholder_A:
            st.markdown("Model A disabled.")

    # Model B
    if use_model_B:
        predsB = predict_B(display_img)
        with placeholder_B:
            st.markdown("**Model B — GeoGuessr-55**")
            for label, score in predsB:
                st.markdown(f"<div class='small-text'>{label}: {score:.1f}%</div>", unsafe_allow_html=True)
            st.bar_chart({'score': [s for _, s in predsB]}, height=120)
    else:
        with placeholder_B:
            st.markdown("Model B disabled.")

# -------------------------
# Mode Logic
# -------------------------
if mode == "Random Local Image (auto)":
    if "random_running" not in st.session_state:
        st.session_state.random_running = False

    if run:
        st.session_state.random_running = True

    if st.session_state.random_running:
        img, chosen_path = get_random_local_image()
        if img:
            country = extract_country_from_path(chosen_path)
            st.caption(f"📁 Random file: {chosen_path}")
            if country:
                st.subheader(f"Country folder: **{country}**")

            render_predictions(img)

        time.sleep(10)
        st.rerun()

elif mode == "Live screenshot (local)":
    if start:
        try:
            while True:
                screenshot = pyautogui.screenshot()
                render_predictions(screenshot.convert("RGB"))
                time.sleep(interval)
        except:
            st.warning("Stopped.")

else:
    if img is not None:
        if chosen_path:
            country = extract_country_from_path(chosen_path)
            st.caption(f"📁 Image path: {chosen_path}")
            if country:
                st.subheader(f"Country folder: **{country}**")

        render_predictions(img)
    else:
        st.info("Upload or select an image to show predictions.")

# Footer
st.markdown("---")
st.caption(f"Using device: **{device}**")

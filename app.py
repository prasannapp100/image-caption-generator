import os
import pickle
import numpy as np
import streamlit as st
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "image_caption_model.keras"   # your trained model
TOKENIZER_PATH = "tokenizer.pkl"           # saved tokenizer
MAX_LEN = 34                               # replace with your training max caption length
FEATURE_DIM = 2048
IMG_SIZE = (299, 299)

st.set_page_config(page_title="AI Caption Generator", page_icon="🖼️", layout="wide")

# -----------------------------
# Custom CSS for styling
# -----------------------------
st.markdown(
    """
    <style>
    .title {
        font-size:40px !important;
        color:#2E86C1;
        text-align:center;
        font-weight:700;
        margin-bottom:0px;
    }
    .subtitle {
        font-size:18px !important;
        color:#555;
        text-align:center;
        margin-top:0px;
        margin-bottom:30px;
    }
    .caption-box {
        padding:15px;
        border-radius:10px;
        background-color:#F4F6F7;
        border:1px solid #D5DBDB;
        margin-top:20px;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_caption_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_extractor():
    base_model = InceptionV3(weights="imagenet")
    return Model(base_model.input, base_model.layers[-2].output)

model = load_caption_model()
tokenizer = load_tokenizer()
feature_extractor = load_feature_extractor()

# -----------------------------
# Utilities
# -----------------------------
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def extract_features(img: Image.Image) -> np.ndarray:
    arr = preprocess_image(img)
    features = feature_extractor.predict(arr, verbose=0).flatten()
    return features

def generate_caption(photo_feats: np.ndarray, max_len: int = MAX_LEN) -> str:
    in_text = "start"
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo_feats.reshape(1, FEATURE_DIM), seq], verbose=0)
        yhat_idx = int(np.argmax(yhat))
        word = tokenizer.index_word.get(yhat_idx, None)
        if word is None:
            break
        in_text += " " + word
        if word == "end":
            break
    return in_text.replace("start", "").replace("end", "").strip()

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("<p class='title'>🖼️ Smart Image Caption Generator</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Dil se dekho toh har tasveer bolti hai</p>", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Settings")
st.sidebar.info("💡 Tip: Try uploading photos with people, animals, or objects for best results.")
decoding = st.sidebar.radio("Decoding Strategy", ["Greedy (default)", "Beam Search"])

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns([1,1])

with col1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file is not None:
        with st.spinner("✨ Generating caption..."):
            photo_feats = extract_features(img)
            caption = generate_caption(photo_feats)

        if caption:
            st.markdown(
                f"""
                <div class="caption-box">
                    <h4 style="color:#117A65;">Generated Caption</h4>
                    <p style="font-size:18px;">{caption}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No caption could be generated. Try another image.")

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <hr>
    <div style='text-align:center; color:gray; font-size:14px;'>
    Bade bade apps mein aisi chhoti chhoti captions hoti rehti hain😂😂.
    </div>
    """,
    unsafe_allow_html=True
)

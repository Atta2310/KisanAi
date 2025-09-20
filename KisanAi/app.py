# app.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests

# -------------------
# CONFIGURATION
# -------------------
st.set_page_config(page_title="KisanAI", page_icon="🌾", layout="wide")
st.title("🌾 KisanAI - Crop Disease & Region Predictor")

MODEL_PATH = "models/phase2_model.h5"
IMAGE_SIZE = (224, 224)  # Update as per your model

# -------------------
# LOAD MODEL
# -------------------
@st.cache_resource(show_spinner=True)
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}")
        return None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("Model loaded successfully!")
    return model

model = load_model()

if model:
    st.subheader("Model Summary")
    st.text(model.summary())

# -------------------
# IMAGE PREDICTION
# -------------------
st.sidebar.header("Crop Disease Prediction")
uploaded_file = st.sidebar.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if

# app.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from gtts import gTTS
from deep_translator import GoogleTranslator
from io import BytesIO

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="🌾 KisanAI",
    page_icon="🌱",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5fff0;
    }
    .stButton>button {
        background-color: #5cb85c;
        color: white;
        font-weight: bold;
    }
    .stSelectbox, .stTextInput {
        background-color: #e6ffe6;
    }
    .stSidebar {
        background-color: #eaffea;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 KisanAI – Crop Disease Predictor & Advisor")

MODEL_PATH = "models/phase2_model.h5"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = {
    0: "Rice - Bacterial leaf blight",
    1: "Rice - Brown spot",
    2: "Rice - Healthy",
    3: "Wheat - Leaf rust",
    4: "Wheat - Stem rust",
    5: "Wheat - Healthy",
    6: "Sugarcane - Red rot",
    7: "Sugarcane - Top shoot borer",
    8: "Sugarcane - Healthy"
}

SUPPORTED_LANGUAGES = {
    "English": "en",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Sindhi": "sd",
    "Pashto": "ps"
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    full_path = os.path.join(os.getcwd(), MODEL_PATH)
    if not os.path.exists(full_path):
        st.error(f"❌ Model not found at {full_path}")
        return None
    try:
        model = tf.keras.models.load_model(full_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -----------------------------
# TRANSLATION FUNCTION
# -----------------------------
def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception as e:
        return text

# -----------------------------
# AUDIO GENERATION
# -----------------------------
def generate_audio(text, lang_code):
    try:
        tts = gTTS(text=text, lang=lang_code)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes
    except Exception as e:
        st.warning("Could not generate audio.")
        return None

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image, model):
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Check for multiple inputs
    if len(model.inputs) == 2:
        second_shape = model.inputs[1].shape
        dummy_input2 = np.zeros((1,) + tuple(second_shape[1:]), dtype=np.float32)
        prediction = model.predict([img_array, dummy_input2])
    else:
        prediction = model.predict(img_array)

    class_idx = np.argmax(prediction)
    return CLASS_NAMES.get(class_idx, "Unknown")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Upload & Settings")
language = st.sidebar.selectbox("Select Output Language", list(SUPPORTED_LANGUAGES.keys()))
uploaded_file = st.sidebar.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

# -----------------------------
# MAIN DISPLAY
# -----------------------------
if uploaded_file is not None:
    st.image(uploaded_file, caption="🌿 Uploaded Image", use_column_width=True)
    
    image = Image.open(uploaded_file)
    
    if model:
        with st.spinner("🔍 Analyzing..."):
            prediction = predict(image, model)
            lang_code = SUPPORTED_LANGUAGES[language]

            translated = translate_text(prediction, lang_code)
            st.success(f"🧪 Disease Prediction: **{translated}**")

            # Audio output
            audio_bytes = generate_audio(translated, lang_code)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
    else:
        st.error("🚫 Model not loaded properly.")
else:
    st.info("📤 Please upload an image to get started.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
👨‍🌾 **KisanAI** helps farmers identify crop diseases with the power of AI.  
Built with ❤️ using TensorFlow, Streamlit, and Google Translate.
""")

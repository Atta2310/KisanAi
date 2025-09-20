import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from gtts import gTTS
from deep_translator import GoogleTranslator
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="🌾 KisanAI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define MODEL_PATH first
MODEL_PATH = "models/phase2_model.h5"


# Debugging info about model file presence
st.write("Current working directory:", os.getcwd())
st.write("Looking for model at:", os.path.abspath(MODEL_PATH))
st.write("Does model exist?", os.path.exists(MODEL_PATH))

IMAGE_SIZE = (224, 224)
API_KEY = "cfb4cdf28f32906b67d0e60ca283e88e"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("❌ Model file not found. Upload `model.h5` to run predictions.")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# CLASS NAMES (update if needed)
# -----------------------------
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

# -----------------------------
# FUNCTIONS
# -----------------------------
def predict_disease(image):
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    return CLASS_NAMES[class_idx]

def translate_text(text, language):
    if language.lower() != "english":
        try:
            return GoogleTranslator(source="auto", target=language.lower()).translate(text)
        except:
            return text
    return text

def crop_region(crop_name):
    # Static mapping for Pakistan regions
    regions = {
        "Rice": ["Sindh: Hyderabad, Sukkur", "Punjab: Lahore, Faisalabad"],
        "Wheat": ["Punjab: Multan, Sahiwal", "KPK: Peshawar"],
        "Sugarcane": ["Sindh: Nawabshah", "Punjab: Bahawalpur"]
    }
    return regions.get(crop_name.title(), ["No suitable region found."])

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("KisanAI 🌾")
st.sidebar.markdown("AI-powered crop disease detector & region advisor.")

tab = st.sidebar.radio("Select Feature", ["Disease Detector", "Crop Region Expert"])
language = st.sidebar.selectbox("Select Language", ["English", "Urdu", "Sindhi", "Punjabi"])

# -----------------------------
# MAIN APP
# -----------------------------
st.title("🌱 KisanAI - Crop Advisor")

if tab == "Disease Detector":
    st.header("📸 Upload Crop Image for Disease Detection")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if model:
            with st.spinner("Analyzing..."):
                disease = predict_disease(image)
                disease_translated = translate_text(disease, language)
                st.success(f"Prediction: {disease_translated}")
        else:
            st.error("Model not loaded!")

elif tab == "Crop Region Expert":
    st.header("🌾 Crop Region Advisor")
    crop_name = st.text_input("Enter Crop Name (Rice/Wheat/Sugarcane):")
    
    if st.button("Find Suitable Regions"):
        if crop_name.strip() != "":
            regions = crop_region(crop_name)
            st.info(f"Suitable regions for {crop_name.title()}:")
            for region in regions:
                st.write(f"- {region}")
        else:
            st.error("Please enter a valid crop name!")

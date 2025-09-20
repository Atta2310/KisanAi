import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator
import os

# -----------------------------
# CONFIGURATION & STYLING
# -----------------------------

st.set_page_config(
    page_title="🌾 KisanAI - Crop Advisor",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    /* General font & background */
    body, .css-18e3th9 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: #f0f7f4;
    }
    /* Title style */
    .main > div > div > div > h1 {
        color: #2c6e49;
        font-weight: 700;
        font-size: 3rem;
        margin-bottom: 0.25rem;
    }
    /* Sidebar */
    .css-1d391kg, .css-1v3fvcr {
        background-color: #2c6e49;
        color: white;
        padding: 1rem 1.5rem;
    }
    /* Sidebar widgets */
    .css-14xtw13, .css-1avcm0n, .css-1v0mbdj {
        color: #e6f0ea;
    }
    /* Buttons */
    div.stButton > button {
        background-color: #3a8d56;
        color: white;
        border-radius: 10px;
        height: 3rem;
        width: 100%;
        font-weight: 600;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #276637;
        cursor: pointer;
    }
    /* File uploader */
    .css-1pahdxg-control {
        border-radius: 10px;
        border: 1.5px solid #3a8d56;
    }
    /* Prediction text */
    .stSuccess {
        font-size: 1.5rem;
        color: #276637;
        font-weight: 700;
    }
    .stWarning {
        color: #cc3d3d;
        font-weight: 700;
    }
    </style>
    """, unsafe_allow_html=True
)

MODEL_PATH = "models/phase2_model.h5"
IMAGE_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    st.info(f"Loading model from: `{MODEL_PATH}`")
    if not os.path.exists(MODEL_PATH):
        st.warning("❌ Model file not found. Upload `phase2_model.h5` to `models` folder.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# -----------------------------
# CLASS NAMES
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
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Check model inputs
    if not model:
        return None

    if len(model.inputs) == 1:
        preds = model.predict(img_array)
    else:
        # Prepare dummy second input as zeros with correct shape
        second_input_shape = model.inputs[1].shape  # (None, ...)
        dummy_input = np.zeros((1,) + tuple(second_input_shape[1:]), dtype=np.float32)
        preds = model.predict([img_array, dummy_input])
    class_idx = np.argmax(preds, axis=1)[0]
    return CLASS_NAMES.get(class_idx, "Unknown")

def translate_text(text, language):
    if language.lower() != "english":
        try:
            return GoogleTranslator(source="auto", target=language.lower()).translate(text)
        except:
            return text
    return text

def crop_region(crop_name):
    regions = {
        "Rice": ["Sindh: Hyderabad, Sukkur", "Punjab: Lahore, Faisalabad"],
        "Wheat": ["Punjab: Multan, Sahiwal", "KPK: Peshawar"],
        "Sugarcane": ["Sindh: Nawabshah", "Punjab: Bahawalpur"]
    }
    return regions.get(crop_name.title(), ["No suitable region found."])

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🌾 KisanAI")
    st.markdown("AI-powered crop disease detector & region advisor.")
    tab = st.radio("Select Feature", ["Disease Detector", "Crop Region Expert"])
    language = st.selectbox("Select Language", ["English", "Urdu", "Sindhi", "Punjabi"])

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
                if disease:
                    st.success(f"Prediction: {disease_translated}")
                else:
                    st.error("Prediction failed.")
        else:
            st.error("Model not loaded!")

    # Show model input details
    if model:
        st.markdown("---")
        st.subheader("Model Info")
        st.write(f"Model expects {len(model.inputs)} input(s):")
        for i, inp in enumerate(model.inputs):
            st.write(f"- Input {i+1} shape: {inp.shape}")

elif tab == "Crop Region Expert":
    st.header("🌾 Crop Region Advisor")
    crop_name = st.text_input("Enter Crop Name (Rice/Wheat/Sugarcane):")

    if st.button("Find Suitable Regions"):
        if crop_name.strip():
            regions = crop_region(crop_name)
            st.info(f"Suitable regions for {crop_name.title()}:")
            for region in regions:
                st.write(f"• {region}")
        else:
            st.error("Please enter a valid crop name!")


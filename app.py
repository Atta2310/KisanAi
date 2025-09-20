import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained Phase 2 model
MODEL_PATH = "models/phase2_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Define class labels (example: update these to match your dataset)
CLASS_NAMES = [
    "Healthy", 
    "Disease 1", 
    "Disease 2", 
    "Disease 3"
]

st.set_page_config(page_title="🌾 KisanAI - Crop Disease Detector", layout="wide")

# UI Header
st.title("🌾 KisanAI - Smart Crop Disease Detection")
st.markdown(
    """
    **Expert System trained with 30+ years of agricultural knowledge**  
    Upload a leaf image and KisanAI will predict if it’s healthy or diseased.
    """
)

# File uploader
uploaded_file = st.file_uploader("📸 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))  # adjust size to your training setup
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    score = np.max(prediction)
    class_idx = np.argmax(prediction)
    result = CLASS_NAMES[class_idx]

    st.subheader("🔍 Prediction Result")
    st.success(f"**{result}** with {score*100:.2f}% confidence")

    # Expert advice section
    st.subheader("👨‍🌾 Expert Advice (Based on 30 years of agricultural knowledge)")
    if result == "Healthy":
        st.info("✅ Your crop looks healthy! Keep monitoring regularly and ensure proper watering and fertilization.")
    else:
        st.warning(f"⚠️ Detected {result}.")
        st.write("👉 Recommended steps:")
        st.write("- Remove affected leaves to prevent spread")
        st.write("- Use recommended fungicides/pesticides")
        st.write("- Rotate crops to reduce soil-borne pathogens")
        st.write("- Consult a local agricultural extension expert for precise treatment")

st.markdown("---")
st.markdown("🚀 Developed with ❤️ for farmers in Pakistan by **KisanAI**")
# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests

# -------------------------
# Constants
# -------------------------
MODEL_PATH = "models/phase2_model.h5"
  # Make sure this is in the same folder
IMAGE_SIZE = (224, 224)         # Adjust according to your model
API_KEY = "YOUR_OPENWEATHER_API_KEY"

# -------------------------
# Load Model
# -------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

st.set_page_config(
    page_title="KisanAI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("KisanAI 🌱")
page = st.sidebar.radio("Navigation", ["Home", "Predict Crop", "Best Region API"])

# -------------------------
# Home Page
# -------------------------
if page == "Home":
    st.markdown("<h1 style='text-align: center; color: green;'>🌾 KisanAI</h1>", unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center;'>
        AI-powered crop prediction & best region recommendation in Pakistan.
        </p>
    """, unsafe_allow_html=True)

    st.image("assets/logo.png", use_column_width=True)

# -------------------------
# Crop Prediction Page
# -------------------------
elif page == "Predict Crop":
    st.header("Predict Crop Disease/Type")
    
    uploaded_file = st.file_uploader("Upload Crop Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Process image
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # -------------------------
        # Prepare dummy second input if model expects 2 inputs
        # -------------------------
        if len(model.inputs) == 2:
            input2_shape = model.inputs[1].shape  # e.g., (None, 7,7,1280)
            dummy_input2 = np.zeros(shape=(1,) + tuple(input2_shape[1:]), dtype=np.float32)
            preds = model.predict([img_array, dummy_input2])
        else:
            preds = model.predict(img_array)
        
        # Display prediction
        pred_class = np.argmax(preds, axis=1)
        st.success(f"Predicted Class ID: {pred_class[0]}")

# -------------------------
# Best Region API Page
# -------------------------
elif page == "Best Region API":
    st.header("Find Best Region for Crop 🌾")
    crop_name = st.text_input("Enter Crop Name (e.g., Wheat, Rice):")
    
    if st.button("Check Best Regions"):
        if crop_name:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={crop_name}&appid={API_KEY}"
            try:
                response = requests.get(url)
                data = response.json()
                
                if data.get("cod") == 200:
                    st.success(f"Weather data received for {crop_name} region!")
                    st.json(data)
                else:
                    st.error(f"Error fetching data: {data.get('message')}")
            except Exception as e:
                st.error(f"API Error: {e}")
        else:
            st.warning("Please enter a crop name.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ by KisanAI</p>", unsafe_allow_html=True)

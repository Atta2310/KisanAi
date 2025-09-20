import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile

# Assume model loading code here (fix your model inputs issue first)
MODEL_PATH = "models/phase2_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = {
    0: "Rice - Bacterial leaf blight",
    1: "Rice - Brown spot",
    2: "Rice - Healthy",
    # etc...
}

DISEASE_SOLUTIONS = {
    "Rice - Bacterial leaf blight": "Use resistant varieties, apply recommended bactericides.",
    "Rice - Brown spot": "Improve soil health and apply fungicides.",
    # Add solution text for each disease
}

LANG_CODE_MAP = {
    "English": "en",
    "Urdu": "ur",
    "Punjabi": "pa",
    "Sindhi": "sd",
    "Pashto": "ps"
}

def translate_text(text, target_lang):
    if target_lang != "English":
        try:
            return GoogleTranslator(source='auto', target=target_lang.lower()).translate(text)
        except Exception:
            return text
    return text

def text_to_speech(text, lang_code):
    tts = gTTS(text=text, lang=lang_code)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp_file.name)
    return tmp_file.name

st.title("🌱 KisanAI Crop Disease Detector")

language = st.selectbox("Select Language", list(LANG_CODE_MAP.keys()))
uploaded_file = st.file_uploader("Upload crop image...", type=["jpg", "png", "jpeg"])

if uploaded_file and st.button("Predict"):
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # For multi-input model, add dummy second input (adjust as needed)
    # dummy_input = np.zeros((1, 7, 7, 1280), dtype=np.float32)
    # preds = model.predict([img_array, dummy_input])

    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds)
    disease_name = CLASS_NAMES[pred_class_idx]
    solution = DISEASE_SOLUTIONS.get(disease_name, "No solution available.")

    disease_translated = translate_text(disease_name, language)
    solution_translated = translate_text(solution, language)
    st.success(f"Disease: {disease_translated}")
    st.info(f"Solution: {solution_translated}")

    # Text to speech in selected language
    lang_code = LANG_CODE_MAP[language]
    combined_text = f"{disease_translated}. {solution_translated}"
    audio_file = text_to_speech(combined_text, lang_code)
    st.audio(audio_file)

    # Clean up temp audio file after use
    os.remove(audio_file)

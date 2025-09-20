# gradio_ui_full.py

import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np
from gtts import gTTS
from io import BytesIO
from googletrans import Translator
import requests
from utils.helper_functions import preprocess_image, generate_tts, LANG_CODES

# ---------------- CONFIG ----------------
MODEL_PATH = "models/phase2_model.h5"
WEATHER_API_KEY = "cfb4cdf28f32906b67d0e60ca283e88e"
LABELS = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro', 'data', 'wheat_leaf']

# ---------------- LOAD MODEL ----------------
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    print("Error loading model:", e)
    model = None

translator = Translator()

# ---------------- PREDICTION FUNCTION ----------------
def predict_disease(image, language):
    if model is None:
        return "Model not loaded", None, "Model not loaded", None

    img_array = preprocess_image(image)

    # Check for multi-input models
    if isinstance(model.input, list) and len(model.input) > 1:
        dummy_input = np.zeros((1,) + tuple(model.inputs[1].shape[1:]), dtype=np.float32)
        pred = model.predict([img_array, dummy_input])
    else:
        pred = model.predict(img_array)

    class_idx = np.argmax(pred, axis=1)[0]
    label = LABELS[class_idx]

    # Translate label and advice
    translated_label = translator.translate(label, dest=LANG_CODES[language]).text
    label_audio = generate_tts(translated_label, LANG_CODES[language])

    advice_dict = {
        'Bacterialblight': "Use certified seeds and practice crop rotation.",
        'Blast': "Apply fungicides and remove infected plant residues.",
        'Brownspot': "Maintain proper irrigation and avoid excessive nitrogen.",
        'Tungro': "Control vector insects and remove infected plants.",
        'data': "Check dataset or model for correct input.",
        'wheat_leaf': "Apply fungicides and monitor leaf health regularly."
    }

    advice_text = advice_dict.get(label, "Follow standard crop care practices.")
    translated_advice = translator.translate(advice_text, dest=LANG_CODES[language]).text
    advice_audio = generate_tts(translated_advice, LANG_CODES[language])

    return translated_label, label_audio, translated_advice, advice_audio

# ---------------- GRADIO INTERFACE ----------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌾 KisanAI - Crop Disease Detection")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            language_input = gr.Dropdown(list(LANG_CODES.keys()), label="Select Language", value="en")
            submit_btn = gr.Button("Predict")
        with gr.Column():
            label_output = gr.Textbox(label="Predicted Disease")
            label_audio_output = gr.Audio(label="Disease Name Audio")
            advice_output = gr.Textbox(label="Advice")
            advice_audio_output = gr.Audio(label="Advice Audio")

    submit_btn.click(
        fn=predict_disease,
        inputs=[image_input, language_input],
        outputs=[label_output, label_audio_output, advice_output, advice_audio_output]
    )

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    demo.launch(share=True)

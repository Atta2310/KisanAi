import numpy as np
from PIL import Image
from gtts import gTTS
from io import BytesIO
from googletrans import Translator

# ---------------- TRANSLATOR ----------------
translator = Translator()

# ---------------- LANGUAGE CODES ----------------
LANG_CODES = {
    "English": "en",
    "Urdu": "ur",
    "Sindhi": "sd",
    "Punjabi": "pa",
    "Pashto": "ps"
}

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image):
    """
    Resize image to 224x224, convert to RGB, normalize and expand dims.
    """
    img = image.convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    return img_array

# ---------------- TEXT-TO-SPEECH ----------------
def generate_tts(text, lang):
    """
    Generate speech audio from text. Returns a BytesIO object for Gradio audio.
    """
    try:
        tts = gTTS(text, lang=lang, slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        print("TTS Error:", e)
        return None

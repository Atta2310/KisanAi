
import numpy as np
from PIL import Image
from gtts import gTTS
from io import BytesIO
from googletrans import Translator

translator = Translator()

LANG_CODES = {
    "English": "en",
    "Urdu": "ur",
    "Sindhi": "sd",
    "Punjabi": "pa",
    "Pashto": "ps"
}

def preprocess_image(image):
    img = image.resize((224,224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    return img_array

def generate_tts(text, lang):
    tts = gTTS(text, lang=lang, slow=False)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

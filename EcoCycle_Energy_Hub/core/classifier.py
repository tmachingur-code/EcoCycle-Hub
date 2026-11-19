import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config.settings import CLASS_NAMES, IMAGE_SIZE

def classify_crop(image: Image.Image, model):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(array, axis=0))
    probabilities = model.predict(batch, verbose=0)[0]
    index = int(np.argmax(probabilities))
    return {
        "material": CLASS_NAMES[index],
        "confidence": float(probabilities[index]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
    }

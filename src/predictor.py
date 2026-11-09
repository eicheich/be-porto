import io
import time
import os
import numpy as np
from PIL import Image
from src.config import MODEL_PATH, TARGET_SIZE, CLASS_NAMES

# ==============================================================================
# SINGLETON LAZY-LOAD FUNCTION
# Model HANYA dimuat saat get_model() dipanggil pertama kali, bukan saat modul di-import!
# ==============================================================================
_classifier_model = None

def get_model():
    """Singleton getter untuk memuat model klasifikasi gambar secara lazy."""
    global _classifier_model
    if _classifier_model is None:
        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        try:
            import keras
            print(f"[INFO] [Lazy Load] Memuat model Keras dari: {MODEL_PATH}")
            _classifier_model = keras.models.load_model(MODEL_PATH)
            print("[INFO] Model Keras berhasil dimuat ke memori.")
        except Exception as e:
            try:
                from tensorflow import keras as tf_keras
                print(f"[INFO] [Lazy Load] Mencoba memuat via tf.keras dari: {MODEL_PATH}")
                _classifier_model = tf_keras.models.load_model(MODEL_PATH)
                print("[INFO] Model Keras berhasil dimuat via tf.keras.")
            except Exception as err:
                print(f"[ERROR] Gagal memuat model klasifikasi gambar: {err}")
                _classifier_model = None
    return _classifier_model


def preprocess_image(file_stream) -> np.ndarray:
    """
    Membaca stream gambar, mengubah ke mode RGB, resize ke target_size (150, 150),
    dan melakukan normalisasi skala 1/255.0.
    """
    image = Image.open(file_stream)
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(file_stream) -> dict:
    """Menjalankan inferensi dengan mengambil model via get_model()."""
    model = get_model()
    if model is None:
        raise RuntimeError(f"Model klasifikasi gambar di '{MODEL_PATH}' tidak tersedia atau gagal dimuat.")

    start_time = time.time()
    processed_img = preprocess_image(file_stream)
    raw_probs = model.predict(processed_img, verbose=0)[0]

    top_idx = int(np.argmax(raw_probs))
    predicted_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"class_{top_idx}"
    confidence = float(raw_probs[top_idx])

    class_probabilities = {}
    for idx, prob in enumerate(raw_probs):
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{idx}"
        class_probabilities[name] = round(float(prob), 4)

    duration_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": class_probabilities,
        "execution_time_ms": duration_ms
    }


class ImagePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path

    @property
    def model(self):
        return get_model()

    def get_model(self):
        return get_model()

    def is_ready(self) -> bool:
        return _classifier_model is not None or os.path.exists(self.model_path)

    def preprocess_image(self, file_stream) -> np.ndarray:
        return preprocess_image(file_stream)

    def predict(self, file_stream) -> dict:
        return predict(file_stream)


_image_predictor_instance = None

def get_image_predictor() -> ImagePredictor:
    global _image_predictor_instance
    if _image_predictor_instance is None:
        _image_predictor_instance = ImagePredictor()
    return _image_predictor_instance

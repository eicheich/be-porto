import io
import time
import os
import numpy as np
from PIL import Image
from src.config import MODEL_PATH, TARGET_SIZE, CLASS_NAMES

class ImagePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None  # Model TIDAK dimuat saat inisialisasi (Lazy Loading)

    def get_model(self):
        """Getter model dengan mekanisme Lazy Loading."""
        if self.model is None:
            self._load_model()
        return self.model

    def _load_model(self):
        """Memuat model hanya saat dibutuhkan pertama kali."""
        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        try:
            import keras
            print(f"[INFO] [Lazy Load] Memuat model Keras dari: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("[INFO] Model Keras berhasil dimuat ke memori.")
        except Exception as e:
            try:
                from tensorflow import keras as tf_keras
                print(f"[INFO] [Lazy Load] Mencoba memuat via tf.keras dari: {self.model_path}")
                self.model = tf_keras.models.load_model(self.model_path)
                print("[INFO] Model Keras berhasil dimuat via tf.keras.")
            except Exception as err:
                print(f"[ERROR] Gagal memuat model klasifikasi gambar: {err}")
                self.model = None

    def is_ready(self) -> bool:
        """Cek apakah model sudah dimuat atau file model ada di disk."""
        return self.model is not None or os.path.exists(self.model_path)

    def preprocess_image(self, file_stream) -> np.ndarray:
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

    def predict(self, file_stream) -> dict:
        """Menjalankan inferensi pada stream gambar (Lazy load jika belum dimuat)."""
        model = self.get_model()
        if model is None:
            raise RuntimeError(f"Model klasifikasi gambar di '{self.model_path}' tidak tersedia atau gagal dimuat.")

        start_time = time.time()
        processed_img = self.preprocess_image(file_stream)
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

# Lazy Singleton Instance (tidak memuat bobot model saat modul di-import)
_image_predictor_instance = None

def get_image_predictor() -> ImagePredictor:
    global _image_predictor_instance
    if _image_predictor_instance is None:
        _image_predictor_instance = ImagePredictor()
    return _image_predictor_instance

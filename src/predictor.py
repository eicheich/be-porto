import io
import time
import numpy as np
from PIL import Image
from src.config import MODEL_PATH, TARGET_SIZE, CLASS_NAMES

class ImagePredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            # Import di dalam method untuk fleksibilitas backend keras/tensorflow
            import keras
            print(f"[INFO] Memuat model Keras dari: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("[INFO] Model berhasil dimuat ke memori.")
        except Exception as e:
            try:
                from tensorflow import keras as tf_keras
                print(f"[INFO] Mencoba memuat model via tf.keras dari: {self.model_path}")
                self.model = tf_keras.models.load_model(self.model_path)
                print("[INFO] Model berhasil dimuat via tf.keras.")
            except Exception as err:
                print(f"[ERROR] Gagal memuat model: {err}")
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess_image(self, file_stream) -> np.ndarray:
        """
        Membaca stream gambar, mengubah ke mode RGB, resize ke target_size (150, 150),
        dan melakukan normalisasi skala 1/255.0.
        """
        image = Image.open(file_stream)
        
        # Pastikan format gambar selalu 3-channel RGB (mengatasi RGBA atau grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize gambar ke dimensi input model
        image = image.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

        # Konversi ke NumPy array dan normalisasi pixel 0 - 1
        img_array = np.asarray(image, dtype=np.float32) / 255.0

        # Tambahkan dimensi batch: (150, 150, 3) -> (1, 150, 150, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict(self, file_stream) -> dict:
        """
        Menjalankan inferensi pada input gambar.
        Mengembalikan label prediksi, confidence, dan probabilitas per kelas.
        """
        if not self.is_ready():
            raise RuntimeError("Model belum berhasil dimuat.")

        start_time = time.time()
        
        # Preprocessing gambar
        input_tensor = self.preprocess_image(file_stream)

        # Inferensi model
        predictions = self.model.predict(input_tensor, verbose=0)
        raw_probs = predictions[0]

        # Cari kelas dengan probabilitas tertinggi
        top_idx = int(np.argmax(raw_probs))
        predicted_class = CLASS_NAMES[top_idx] if top_idx < len(CLASS_NAMES) else f"class_{top_idx}"
        confidence = float(raw_probs[top_idx])

        # Pemetaan probabilitas seluruh kelas
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

# Instance singleton predictor
predictor = ImagePredictor()


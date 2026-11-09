import io
import os
import base64
import numpy as np
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MNIST_MODEL_PATH = os.path.join(BASE_DIR, "model", "mnist_cnn_model.keras")

class MNISTPredictor:
    def __init__(self, model_path=MNIST_MODEL_PATH):
        self.model_path = model_path
        self.model = None  # Model TIDAK dimuat saat inisialisasi (Lazy Loading)

    def get_model(self):
        """Getter model dengan mekanisme Lazy Loading."""
        if self.model is None:
            self._load_model()
        return self.model

    def _load_model(self):
        """Memuat model MNIST hanya saat dibutuhkan pertama kali."""
        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass

        try:
            import keras
            print(f"[INFO] [Lazy Load] Memuat model MNIST CNN dari: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("[INFO] Model MNIST CNN berhasil dimuat ke memori.")
        except Exception as e:
            try:
                from tensorflow import keras as tf_keras
                print(f"[INFO] [Lazy Load] Mencoba memuat model MNIST via tf.keras dari: {self.model_path}")
                self.model = tf_keras.models.load_model(self.model_path)
                print("[INFO] Model MNIST CNN berhasil dimuat via tf.keras.")
            except Exception as err:
                print(f"[ERROR] Gagal memuat model MNIST: {err}")
                self.model = None

    def is_ready(self) -> bool:
        """Cek apakah model sudah dimuat atau file model ada di disk."""
        return self.model is not None or os.path.exists(self.model_path)

    def preprocess_base64_image(self, base64_str: str) -> np.ndarray:
        """
        1. Bersihkan prefix data:image/...;base64,
        2. Decode base64 menjadi gambar PIL
        3. Konversi gambar transparan (RGBA) ke latar belakang putih jika diperlukan
        4. Ubah ke Grayscale (1 channel)
        5. Resize ke 28x28 pixel
        6. Inversi warna jika latar belakang putih dan tulisan hitam (MNIST = latar hitam, tulisan putih)
        7. Normalisasi pixel (0.0 - 1.0)
        8. Reshape ke (1, 28, 28, 1)
        """
        if not base64_str or not isinstance(base64_str, str):
            raise ValueError("Input base64 string tidak valid atau kosong.")

        # 1. Bersihkan prefix data:image/...;base64,
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]

        # 2. Decode string base64
        try:
            img_bytes = base64.b64decode(base64_str.strip())
            image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise ValueError(f"Gagal mendecode base64 menjadi gambar: {str(e)}")

        # 3. Tangani alpha channel (canvas transparan browser)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            image = Image.alpha_composite(bg, image)

        # 4. Ubah gambar menjadi Grayscale (1 channel)
        image = image.convert("L")

        # 5. Resize gambar menjadi 28x28 pixel
        image = image.resize((28, 28), Image.Resampling.LANCZOS)

        # 6. Inversi warna jika canvas berlatar belakang putih (MNIST = latar hitam, tulisan putih)
        np_temp = np.array(image, dtype=np.float32)
        if np.mean(np_temp) > 127:
            image = ImageOps.invert(image)

        # 7. Normalisasi pixel / 255.0
        img_array = np.array(image, dtype=np.float32) / 255.0

        # 8. Reshape menjadi (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1)
        return img_array

    def predict(self, base64_str: str) -> dict:
        """Menjalankan inferensi pada gambar digit base64 (Lazy load jika belum dimuat)."""
        model = self.get_model()
        if model is None:
            raise RuntimeError(f"Model MNIST CNN di '{self.model_path}' tidak tersedia atau gagal dimuat.")

        input_tensor = self.preprocess_base64_image(base64_str)
        predictions = model.predict(input_tensor, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        return {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in predictions]
        }

# Lazy Singleton Instance (tidak memuat bobot model saat modul di-import)
_mnist_predictor_instance = None

def get_mnist_predictor() -> MNISTPredictor:
    global _mnist_predictor_instance
    if _mnist_predictor_instance is None:
        _mnist_predictor_instance = MNISTPredictor()
    return _mnist_predictor_instance

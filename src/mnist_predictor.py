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
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            import keras
            print(f"[INFO] Memuat model MNIST CNN dari: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("[INFO] Model MNIST CNN berhasil dimuat ke memori.")
        except Exception as e:
            try:
                from tensorflow import keras as tf_keras
                print(f"[INFO] Mencoba memuat model MNIST via tf.keras dari: {self.model_path}")
                self.model = tf_keras.models.load_model(self.model_path)
                print("[INFO] Model MNIST CNN berhasil dimuat via tf.keras.")
            except Exception as err:
                print(f"[ERROR] Gagal memuat model MNIST: {err}")
                self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def preprocess_base64_image(self, base64_str: str) -> np.ndarray:
        """
        1. Bersihkan prefix data:image/...;base64,
        2. Decode base64 menjadi gambar PIL
        3. Konversi gambar transparan (RGBA) ke latar belakang putih jika diperlukan
        4. Ubah ke Grayscale (1 channel)
        5. Resize ke 28x28 pixel
        6. Inversi warna jika latar belakang putih dan tulisan hitam (agar sesuai format MNIST: latar hitam, tulisan putih)
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
            # Gabungkan gambar di atas latar belakang putih
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
        """
        Menjalankan inferensi dan mengembalikan predicted_digit (0-9) dan confidence score.
        """
        if not self.is_ready():
            raise RuntimeError("Model MNIST CNN belum berhasil dimuat.")

        input_tensor = self.preprocess_base64_image(base64_str)
        predictions = self.model.predict(input_tensor, verbose=0)[0]

        predicted_digit = int(np.argmax(predictions))
        confidence = float(predictions[predicted_digit])

        return {
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 4),
            "probabilities": [round(float(p), 4) for p in predictions]
        }

# Instance singleton
mnist_predictor = MNISTPredictor()


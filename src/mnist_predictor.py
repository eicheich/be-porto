import io
import os
import base64
import threading
import numpy as np
from PIL import Image, ImageOps
from src.config import MNIST_MODEL_PATH, MAX_BASE64_LENGTH

# Protect against decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 10_000_000

# ==============================================================================
# SINGLETON LAZY-LOAD FUNCTION (THREAD-SAFE)
# ==============================================================================
_mnist_model = None
_mnist_lock = threading.Lock()


def get_model():
    """Thread-safe singleton getter untuk memuat model MNIST CNN secara lazy."""
    global _mnist_model
    if _mnist_model is None:
        with _mnist_lock:
            if _mnist_model is None:
                intra_threads = int(os.getenv("TF_NUM_INTRAOP_THREADS", "2"))
                inter_threads = int(os.getenv("TF_NUM_INTEROP_THREADS", "1"))
                try:
                    import tensorflow as tf
                    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
                    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
                except Exception:
                    pass

                try:
                    import keras
                    print(f"[INFO] [Lazy Load] Memuat model MNIST CNN dari: {MNIST_MODEL_PATH}")
                    _mnist_model = keras.models.load_model(MNIST_MODEL_PATH)
                    print("[INFO] Model MNIST CNN berhasil dimuat ke memori.")
                except Exception as e:
                    try:
                        from tensorflow import keras as tf_keras
                        print(f"[INFO] [Lazy Load] Mencoba memuat model MNIST via tf.keras dari: {MNIST_MODEL_PATH}")
                        _mnist_model = tf_keras.models.load_model(MNIST_MODEL_PATH)
                        print("[INFO] Model MNIST CNN berhasil dimuat via tf.keras.")
                    except Exception as err:
                        print(f"[ERROR] Gagal memuat model MNIST: {err}")
                        _mnist_model = None
    return _mnist_model


def preprocess_base64_image(base64_str: str) -> np.ndarray:
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

    if len(base64_str) > MAX_BASE64_LENGTH:
        raise ValueError("Ukuran payload base64 melebihi batas maksimum 2 MB.")

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


def predict(base64_str: str) -> dict:
    """Menjalankan inferensi dengan mengambil model via get_model()."""
    model = get_model()
    if model is None:
        raise RuntimeError(f"Model MNIST CNN di '{MNIST_MODEL_PATH}' tidak tersedia atau gagal dimuat.")

    input_tensor = preprocess_base64_image(base64_str)
    predictions = model.predict(input_tensor, verbose=0)[0]

    predicted_digit = int(np.argmax(predictions))
    confidence = float(predictions[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence, 4),
        "probabilities": [round(float(p), 4) for p in predictions]
    }


class MNISTPredictor:
    def __init__(self, model_path=MNIST_MODEL_PATH):
        self.model_path = model_path

    @property
    def model(self):
        return get_model()

    def get_model(self):
        return get_model()

    def is_ready(self) -> bool:
        return _mnist_model is not None or os.path.exists(self.model_path)

    def preprocess_base64_image(self, base64_str: str) -> np.ndarray:
        return preprocess_base64_image(base64_str)

    def predict(self, base64_str: str) -> dict:
        return predict(base64_str)


_mnist_predictor_instance = None

def get_mnist_predictor() -> MNISTPredictor:
    global _mnist_predictor_instance
    if _mnist_predictor_instance is None:
        _mnist_predictor_instance = MNISTPredictor()
    return _mnist_predictor_instance

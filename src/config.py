import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path Model
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_classification_3class.keras")

# Konfigurasi Model
TARGET_SIZE = (150, 150)
CLASS_NAMES = ["cats", "dogs", "others"]

# Konfigurasi Upload
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

# Konfigurasi Server
HOST = os.getenv("FLASK_HOST", "0.0.0.0")
PORT = int(os.getenv("FLASK_PORT", 5000))
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1", "t")


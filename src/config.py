import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Paths
CATDOG_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_classification_3class.keras")
MNIST_MODEL_PATH = os.path.join(BASE_DIR, "model", "mnist_cnn_model.keras")
HOUSE_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_random_forest.joblib")

# Backward compatibility alias
MODEL_PATH = CATDOG_MODEL_PATH

# Model Settings
TARGET_SIZE = (150, 150)
CLASS_NAMES = ["cats", "dogs", "others"]

# Upload & Security Limits
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB safe limit for portfolio inference
MAX_BASE64_LENGTH = 2 * 1024 * 1024   # ~2 MB base64 string

# Server & CORS Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

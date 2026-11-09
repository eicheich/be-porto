import os
import sys

# 1. Atur Environment Variable Pembatas Resource & Threading
# Wajib diletakkan di baris paling awal sebelum library berat (TensorFlow / Keras) di-import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# 2. Tambahkan root directory ke sys.path secara aman
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 3. Batasi threading TensorFlow pada level konfigurasi (mencegah error pthread_create EAGAIN/code 11)
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

# 4. Bungkus FastAPI ASGI app dengan a2wsgi.ASGIMiddleware agar kompatibel dengan Phusion Passenger (WSGI)
from a2wsgi import ASGIMiddleware
from main import app as fastapi_app

application = ASGIMiddleware(fastapi_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

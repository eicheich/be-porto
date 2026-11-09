import os
import sys

# 1. Atur Environment Variable Pembatas Resource & Threading
# Wajib diletakkan di baris paling awal sebelum library berat di-import
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# 2. Tambahkan root directory ke sys.path secara aman
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# 3. Bungkus FastAPI ASGI app dengan a2wsgi.ASGIMiddleware agar kompatibel dengan Phusion Passenger (WSGI)
# Karena menggunakan Lazy Loading, proses import ini instan (<0.1s) dan bebas dari startup timeout/signal 15
from a2wsgi import ASGIMiddleware
from main import app as fastapi_app

application = ASGIMiddleware(fastapi_app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

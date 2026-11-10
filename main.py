import io
import os
import sys
import threading
from contextlib import asynccontextmanager
from typing import Optional

# 1. Konfigurasi Threading & CPU (Optimasi vCPU Docker Container)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", os.getenv("OMP_NUM_THREADS", "2"))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", os.getenv("TF_NUM_INTRAOP_THREADS", "2"))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", os.getenv("TF_NUM_INTEROP_THREADS", "1"))

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import (
    BASE_DIR,
    CATDOG_MODEL_PATH,
    MNIST_MODEL_PATH,
    HOUSE_MODEL_PATH,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    MAX_BASE64_LENGTH,
    HOST,
    PORT,
    CORS_ORIGINS,
)

# ==============================================================================
# 2. MEKANISME LAZY LOADING SINGLETON GETTERS (THREAD-SAFE)
# Model HANYA dimuat saat request pertama masuk ke endpoint terkait!
# ==============================================================================
_house_model = None
_house_lock = threading.Lock()
_mnist_predictor = None
_classifier_predictor = None


def get_house_model():
    """Thread-safe lazy getter untuk model Random Forest (House Pricing)."""
    global _house_model
    if _house_model is None:
        with _house_lock:
            if _house_model is None:
                if os.path.exists(HOUSE_MODEL_PATH):
                    import joblib
                    print(f"[INFO] [Lazy Load] Memuat model Random Forest dari: {HOUSE_MODEL_PATH}")
                    _house_model = joblib.load(HOUSE_MODEL_PATH)
                    print("[INFO] Model Random Forest berhasil dimuat ke memori.")
                else:
                    print(f"[WARNING] File '{HOUSE_MODEL_PATH}' tidak ditemukan di server.")
    return _house_model


def get_mnist_predictor():
    """Lazy getter untuk MNIST Predictor."""
    global _mnist_predictor
    if _mnist_predictor is None:
        from src.mnist_predictor import get_mnist_predictor as _get_mnist
        _mnist_predictor = _get_mnist()
    return _mnist_predictor


def get_mnist_model():
    """Lazy getter untuk model Keras MNIST."""
    from src.mnist_predictor import get_model as _get_mnist_m
    return _get_mnist_m()


def get_classifier_predictor():
    """Lazy getter untuk Image Predictor (Cats vs Dogs)."""
    global _classifier_predictor
    if _classifier_predictor is None:
        from src.predictor import get_image_predictor as _get_img
        _classifier_predictor = _get_img()
    return _classifier_predictor


def get_classifier_model():
    """Lazy getter untuk model Keras Image Classifier (Cats vs Dogs)."""
    from src.predictor import get_model as _get_img_m
    return _get_img_m()


# ==============================================================================
# 3. FASTAPI APP INITIALIZATION
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup instan tanpa memuat model ke memori (Lazy Loading aktif)."""
    print("[INFO] Server FastAPI berhasil boot instan (Lazy Loading aktif).")
    yield
    print("[INFO] Server FastAPI shutdown.")


app = FastAPI(
    title="Portfolio Machine Learning API",
    description="Production-ready FastAPI backend for ML models with Lazy Loading and container optimization.",
    version="2.0.0",
    lifespan=lifespan
)

# Konfigurasi CORS yang aman
is_wildcard = "*" in CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not is_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 4. SKEMA VALIDASI DATA (Pydantic Models)
# ==============================================================================
class HouseFeatures(BaseModel):
    LB: float = Field(..., gt=0, le=100000, description="Luas Bangunan (m2)", examples=[120.0])
    LT: float = Field(..., gt=0, le=100000, description="Luas Tanah (m2)", examples=[150.0])
    KT: int = Field(..., ge=0, le=100, description="Jumlah Kamar Tidur", examples=[3])
    KM: int = Field(..., ge=0, le=100, description="Jumlah Kamar Mandi", examples=[2])
    GRS: int = Field(..., ge=0, le=100, description="Jumlah Garasi", examples=[1])


class PredictionResponse(BaseModel):
    estimasi_harga: float = Field(..., description="Estimasi nilai harga rumah")
    status: str = Field(..., description="Status prediksi (sukses/gagal)")


class HandwriteRequest(BaseModel):
    image_base64: str = Field(
        ...,
        max_length=MAX_BASE64_LENGTH,
        description="String base64 gambar tulisan tangan dari canvas"
    )


class HandwriteResponse(BaseModel):
    predicted_digit: int = Field(..., description="Angka hasil prediksi (0-9)")
    confidence: float = Field(..., description="Tingkat keyakinan model terhadap prediksi (0.0 - 1.0)")


# Helper validasi ekstensi file
def is_allowed_file(filename: Optional[str]) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# ==============================================================================
# 5. ENDPOINTS
# ==============================================================================
@app.get("/", summary="Health Check Root")
def health_check():
    """Endpoint root ringan tanpa memicu pemuatan model."""
    return {
        "status": "online",
        "service": "Portfolio Machine Learning API",
        "lazy_loading": True,
        "model_files": {
            "house_pricing": os.path.exists(HOUSE_MODEL_PATH),
            "mnist_cnn": os.path.exists(MNIST_MODEL_PATH),
            "image_classifier": os.path.exists(CATDOG_MODEL_PATH),
        }
    }


@app.get("/health", summary="Health Check Status")
def health():
    """Memeriksa status container dan model yang sedang aktif di memori (Azure Probe)."""
    from src.mnist_predictor import _mnist_model
    from src.predictor import _classifier_model
    return {
        "status": "healthy",
        "lazy_loading": True,
        "models_in_memory": {
            "house_model": _house_model is not None,
            "mnist_model": _mnist_model is not None,
            "classifier_model": _classifier_model is not None,
        }
    }


@app.post(
    "/predict-house-pricing",
    response_model=PredictionResponse,
    summary="Prediksi Harga Rumah",
    status_code=status.HTTP_200_OK
)
async def predict_house_pricing(features: HouseFeatures):
    """
    Endpoint utama untuk prediksi harga rumah (Random Forest).
    Inferensi dijalankan di worker thread agar tidak memblokir event loop.
    """
    model = get_house_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File model 'model_random_forest.joblib' belum dimuat atau tidak ditemukan di server."
        )

    try:
        def _run_inference():
            input_data = {
                "LB": [features.LB],
                "LT": [features.LT],
                "KT": [features.KT],
                "KM": [features.KM],
                "GRS": [features.GRS]
            }
            df_input = pd.DataFrame(input_data, columns=["LB", "LT", "KT", "KM", "GRS"])
            raw_prediction = model.predict(df_input)
            return float(raw_prediction[0])

        estimasi_nilai = await run_in_threadpool(_run_inference)
        return PredictionResponse(
            estimasi_harga=estimasi_nilai,
            status="sukses"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses prediksi harga: {str(e)}"
        )


@app.post(
    "/predict-handwrite/",
    response_model=HandwriteResponse,
    summary="Prediksi Angka Tulisan Tangan (MNIST)",
    status_code=status.HTTP_200_OK
)
@app.post(
    "/predict-handwrite",
    response_model=HandwriteResponse,
    include_in_schema=False
)
async def predict_handwrite(payload: HandwriteRequest):
    """
    Endpoint untuk mendeteksi angka tulisan tangan (0-9) dari gambar canvas base64.
    Inferensi dijalankan secara asynchronous di threadpool (non-blocking).
    """
    from src.mnist_predictor import predict as predict_digit

    try:
        result = await run_in_threadpool(predict_digit, payload.image_base64)
        return HandwriteResponse(
            predicted_digit=result["predicted_digit"],
            confidence=result["confidence"]
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan saat memproses prediksi angka: {str(e)}"
        )


@app.post("/predict", summary="Klasifikasi Gambar (Cats vs Dogs) atau Fallback")
async def predict(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Endpoint Klasifikasi Gambar (Cats vs Dogs).
    Menerima file gambar (multipart/form-data) dengan validasi ukuran & ekstensi,
    atau payload JSON base64.
    """
    upload = image or file

    # 1. Penanganan Multipart Form Data
    if upload is not None and upload.filename:
        if not is_allowed_file(upload.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Format file tidak didukung. Format yang diizinkan: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        contents = await upload.read()
        if len(contents) > MAX_CONTENT_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Ukuran file melebihi batas maksimum {MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
            )

        from src.predictor import predict as predict_image
        try:
            stream = io.BytesIO(contents)
            result = await run_in_threadpool(predict_image, stream)
            return {
                "success": True,
                **result
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}"
            )

    # 2. Penanganan JSON Payload (Base64 atau House Pricing Fallback)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}

        if "LB" in body:
            features = HouseFeatures(**body)
            return await predict_house_pricing(features)

        elif "image_base64" in body:
            b64_data = body["image_base64"]
            if not isinstance(b64_data, str) or len(b64_data) > MAX_BASE64_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="String base64 tidak valid atau melebihi batas ukuran 2 MB."
                )

            import base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            try:
                image_bytes = base64.b64decode(b64_data)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Format base64 rusak: {str(e)}"
                )

            if len(image_bytes) > MAX_CONTENT_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Ukuran gambar base64 melebihi batas {MAX_CONTENT_LENGTH // (1024 * 1024)} MB."
                )

            from src.predictor import predict as predict_image
            try:
                stream = io.BytesIO(image_bytes)
                result = await run_in_threadpool(predict_image, stream)
                return {"success": True, **result}
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Terjadi kesalahan saat memproses gambar base64: {str(e)}"
                )

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Harap kirimkan file gambar melalui multipart/form-data dengan key 'image' atau 'file'."
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)

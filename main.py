import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

# 1. Konfigurasi Environment Pembatas Resource & Threading (CloudLinux nproc limit & CPU mode)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 2. MEKANISME LAZY LOADING SINGLETON GETTERS
# Tidak ada pemuatan model machine learning di tingkat modul global saat startup!
# ==============================================================================
_house_model = None
_mnist_predictor = None
_classifier_predictor = None


def get_model_path() -> Optional[str]:
    """Mencari lokasi file model Random Forest di folder model/ atau mode/."""
    candidates = [
        os.path.join(BASE_DIR, "model", "model_random_forest.joblib"),
        os.path.join(BASE_DIR, "mode", "model_random_forest.joblib"),
        "model/model_random_forest.joblib",
        "mode/model_random_forest.joblib",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_house_model():
    """Lazy getter untuk model Random Forest (House Pricing)."""
    global _house_model
    if _house_model is None:
        model_path = get_model_path()
        if model_path:
            import joblib
            print(f"[INFO] [Lazy Load] Memuat model Random Forest dari: {model_path}")
            _house_model = joblib.load(model_path)
            print("[INFO] Model Random Forest berhasil dimuat ke memori.")
        else:
            print("[WARNING] File 'model_random_forest.joblib' tidak ditemukan di server.")
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
    predictor = get_mnist_predictor()
    return predictor.get_model()


def get_classifier_predictor():
    """Lazy getter untuk Image Predictor (Cats vs Dogs)."""
    global _classifier_predictor
    if _classifier_predictor is None:
        from src.predictor import get_image_predictor as _get_img
        _classifier_predictor = _get_img()
    return _classifier_predictor


def get_classifier_model():
    """Lazy getter untuk model Keras Image Classifier (Cats vs Dogs)."""
    predictor = get_classifier_predictor()
    return predictor.get_model()


# ==============================================================================
# 3. FASTAPI APP INITIALIZATION
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup instan tanpa memuat model ke memori (Lazy Loading aktif).
    Menghindari killed by signal 15 / startup timeout pada Phusion Passenger cPanel.
    """
    print("[INFO] Server FastAPI berhasil boot instan (Lazy Loading aktif).")
    yield
    print("[INFO] Server FastAPI shutdown.")


app = FastAPI(
    title="House Pricing & ML API",
    description="Backend API berbasis FastAPI dengan mekanisme Lazy Loading untuk kompatibilitas CloudLinux/cPanel Passenger",
    version="1.2.0",
    lifespan=lifespan
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 4. SKEMA VALIDASI DATA (Pydantic Models)
# ==============================================================================
class HouseFeatures(BaseModel):
    LB: float = Field(..., gt=0, description="Luas Bangunan (harus lebih besar dari 0)", examples=[120.0])
    LT: float = Field(..., gt=0, description="Luas Tanah (harus lebih besar dari 0)", examples=[150.0])
    KT: int = Field(..., ge=0, description="Jumlah Kamar Tidur (minimal 0)", examples=[3])
    KM: int = Field(..., ge=0, description="Jumlah Kamar Mandi (minimal 0)", examples=[2])
    GRS: int = Field(..., ge=0, description="Jumlah Garasi (minimal 0)", examples=[1])


class PredictionResponse(BaseModel):
    estimasi_harga: float = Field(..., description="Estimasi nilai harga rumah")
    status: str = Field(..., description="Status prediksi (sukses/gagal)")


class HandwriteRequest(BaseModel):
    image_base64: str = Field(..., description="String base64 gambar tulisan tangan dari canvas")


class HandwriteResponse(BaseModel):
    predicted_digit: int = Field(..., description="Angka hasil prediksi (0-9)")
    confidence: float = Field(..., description="Tingkat keyakinan model terhadap prediksi (0.0 - 1.0)")


# ==============================================================================
# 5. ENDPOINTS
# ==============================================================================
@app.get("/", summary="Health Check Root")
def health_check():
    """Endpoint kesehatan cepat tanpa memicu pemuatan model ke memori."""
    return {
        "status": "online",
        "message": "FastAPI ML Backend is running (Lazy Loading enabled)",
        "model_files": {
            "house_pricing": os.path.exists(get_model_path() or ""),
            "mnist_cnn": os.path.exists(os.path.join(BASE_DIR, "model", "mnist_cnn_model.keras")),
            "image_classifier": os.path.exists(os.path.join(BASE_DIR, "model", "model_classification_3class.keras")),
        }
    }


@app.get("/health", summary="Health Check Status")
def health():
    """Memeriksa status server dan apakah model sudah dimuat ke memori atau belum."""
    return {
        "status": "healthy",
        "lazy_loading": True,
        "models_in_memory": {
            "house_model": _house_model is not None,
            "mnist_model": _mnist_predictor is not None and _mnist_predictor.model is not None,
            "classifier_model": _classifier_predictor is not None and _classifier_predictor.model is not None,
        }
    }


@app.post(
    "/predict-house-pricing",
    response_model=PredictionResponse,
    summary="Prediksi Harga Rumah",
    status_code=status.HTTP_200_OK
)
def predict_house_pricing(features: HouseFeatures):
    """
    Endpoint utama untuk melakukan prediksi harga rumah.
    Model Random Forest akan dimuat secara lazy saat pertama kali dipanggil.
    """
    model = get_house_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File model 'model_random_forest.joblib' belum dimuat atau tidak ditemukan di server."
        )

    try:
        input_data = {
            "LB": [features.LB],
            "LT": [features.LT],
            "KT": [features.KT],
            "KM": [features.KM],
            "GRS": [features.GRS]
        }
        df_input = pd.DataFrame(input_data, columns=["LB", "LT", "KT", "KM", "GRS"])
        raw_prediction = model.predict(df_input)
        estimasi_nilai = float(raw_prediction[0])

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
def predict_handwrite(payload: HandwriteRequest):
    """
    Endpoint untuk mendeteksi angka tulisan tangan (0-9) dari gambar canvas base64.
    Model MNIST CNN akan dimuat secara lazy saat pertama kali dipanggil.
    """
    predictor = get_mnist_predictor()
    try:
        result = predictor.predict(payload.image_base64)
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
    Menerima file gambar (multipart/form-data) atau payload JSON base64.
    Model Keras akan dimuat secara lazy saat pertama kali dipanggil.
    """
    upload = image or file
    if upload is not None and upload.filename:
        predictor = get_classifier_predictor()
        try:
            contents = await upload.read()
            stream = io.BytesIO(contents)
            result = predictor.predict(stream)
            return {
                "success": True,
                **result
            }
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}"
            )

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if "LB" in body:
            features = HouseFeatures(**body)
            return predict_house_pricing(features)
        elif "image_base64" in body:
            predictor = get_classifier_predictor()
            import base64
            b64_data = body["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            result = predictor.predict(io.BytesIO(image_bytes))
            return {"success": True, **result}

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Harap kirimkan file gambar melalui multipart/form-data dengan key 'image'."
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

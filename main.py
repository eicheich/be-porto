import io
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src.mnist_predictor import mnist_predictor

# Inisialisasi variabel global untuk model
model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lazy / safe import for image classification predictor
try:
    from src.predictor import predictor
except Exception as e:
    print(f"[WARNING] Could not load image predictor: {e}")
    predictor = None


def get_model_path() -> Optional[str]:
    """Mencari lokasi file model di folder mode/ atau model/."""
    candidates = [
        os.path.join(BASE_DIR, "mode", "model_random_forest.joblib"),
        os.path.join(BASE_DIR, "model", "model_random_forest.joblib"),
        "mode/model_random_forest.joblib",
        "model/model_random_forest.joblib",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Event lifespan untuk memuat model saat startup aplikasi."""
    global model
    model_path = get_model_path()
    if model_path:
        try:
            print(f"[INFO] Memuat model Random Forest dari: {model_path}")
            model = joblib.load(model_path)
            print("[INFO] Model Random Forest berhasil dimuat ke memori.")
        except Exception as e:
            print(f"[ERROR] Gagal memuat file model: {e}")
            model = None
    else:
        print("[WARNING] File model 'model_random_forest.joblib' tidak ditemukan di folder mode/ atau model/.")
    yield


app = FastAPI(
    title="House Pricing Prediction API",
    description="Backend API berbasis FastAPI untuk estimasi harga rumah menggunakan model Random Forest Regressor",
    version="1.0.0",
    lifespan=lifespan
)

# Konfigurasi CORS agar frontend dapat mengakses endpoint
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. SKEMA VALIDASI DATA (Pydantic Models)
# ==========================================
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


# ==========================================
# 2. STRUKTUR ENDPOINT
# ==========================================
@app.get("/", summary="Health Check")
def health_check():
    """Endpoint kesehatan sederhana yang mengembalikan status server dan kesiapan model."""
    return {
        "status": "online",
        "message": "House Pricing Prediction API is running",
        "model_loaded": model is not None,
        "features_required": ["LB", "LT", "KT", "KM", "GRS"]
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
    Menerima data HouseFeatures, mengubah menjadi DataFrame pandas,
    dan melakukan prediksi menggunakan model Random Forest.
    """
    global model
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File model 'model_random_forest.joblib' belum dimuat atau tidak ditemukan di server."
        )

    try:
        # Konversi data input Pydantic menjadi pandas DataFrame dengan nama kolom yang sesuai
        input_data = {
            "LB": [features.LB],
            "LT": [features.LT],
            "KT": [features.KT],
            "KM": [features.KM],
            "GRS": [features.GRS]
        }
        df_input = pd.DataFrame(input_data, columns=["LB", "LT", "KT", "KM", "GRS"])

        # Lakukan prediksi menggunakan model
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
    Menerima JSON: {"image_base64": "data:image/png;base64,..." atau "base64_string"}
    Mengembalikan: {"predicted_digit": int, "confidence": float}
    """
    if mnist_predictor is None or not mnist_predictor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model MNIST CNN belum siap atau file 'model/mnist_cnn_model.keras' gagal dimuat."
        )

    try:
        result = mnist_predictor.predict(payload.image_base64)
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


@app.get("/health", summary="Health Check Model")
def health():
    ready = predictor is not None and predictor.is_ready()
    mnist_ready = mnist_predictor is not None and mnist_predictor.is_ready()
    return {
        "status": "healthy" if (ready or mnist_ready or model is not None) else "degraded",
        "model_loaded": ready,
        "house_model_loaded": model is not None,
        "mnist_model_loaded": mnist_ready
    }


# Endpoint untuk Prediksi Gambar (Dog vs Cat) & fallback
@app.post("/predict", summary="Klasifikasi Gambar (Cats vs Dogs) atau Fallback")
async def predict(
    request: Request,
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    upload = image or file
    if upload is not None and upload.filename:
        if predictor is None or not predictor.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model klasifikasi gambar belum siap atau gagal dimuat."
            )
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
            if predictor is None or not predictor.is_ready():
                raise HTTPException(status_code=503, detail="Model belum siap.")
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

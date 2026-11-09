import io
import os
import base64
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from src.config import (
    MODEL_PATH,
    TARGET_SIZE,
    CLASS_NAMES,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    HOST,
    PORT,
    DEBUG
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_house_model = None

def get_house_model():
    global _house_model
    if _house_model is None:
        for candidate in [
            os.path.join(BASE_DIR, "model", "model_random_forest.joblib"),
            os.path.join(BASE_DIR, "mode", "model_random_forest.joblib"),
        ]:
            if os.path.exists(candidate):
                try:
                    _house_model = joblib.load(candidate)
                    print(f"[INFO] [Lazy Load] Model Random Forest berhasil dimuat dari: {candidate}")
                    break
                except Exception as e:
                    print(f"[ERROR] Gagal memuat model Random Forest: {e}")
    return _house_model

def get_image_predictor():
    from src.predictor import get_image_predictor as _get
    return _get()

def get_mnist_predictor():
    from src.mnist_predictor import get_mnist_predictor as _get
    return _get()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Aktifkan CORS untuk mendukung integrasi frontend dari domain/port berbeda
CORS(app, resources={r"/*": {"origins": "*"}})


def is_allowed_file(filename: str) -> bool:
    """Memeriksa ekstensi file yang diizinkan."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    """Halaman indeks API."""
    return jsonify({
        "name": "Image Classification API",
        "description": "Backend API untuk klasifikasi gambar 3 kelas (cats, dogs, others)",
        "endpoints": {
            "GET /health": "Memeriksa status server dan ketersediaan model",
            "POST /predict": "Mengunggah gambar untuk mendapatkan hasil prediksi (multipart/form-data)"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    """
    Endpoint Health Check.
    Memeriksa apakah server berjalan (tidak memicu pemuatan model).
    """
    return jsonify({
        "status": "healthy",
        "lazy_loading": True,
        "models_in_memory": {
            "house_model": _house_model is not None,
            "catdog_model": get_image_predictor().model is not None,
            "mnist_model": get_mnist_predictor().model is not None,
        },
        "model_path": MODEL_PATH,
        "target_classes": CLASS_NAMES
    }), 200


@app.route("/predict-house-pricing", methods=["POST"])
def predict_house_pricing():
    """
    Endpoint Prediksi Harga Rumah (Random Forest).
    Menerima JSON dengan field: LB, LT, KT, KM, GRS.
    """
    model = get_house_model()
    if model is None:
        return jsonify({
            "status": "gagal",
            "error": "Model Random Forest belum dimuat atau tidak ditemukan di server."
        }), 500

    if not request.is_json:
        return jsonify({
            "status": "gagal",
            "error": "Payload harus berformat JSON dengan header Content-Type: application/json."
        }), 400

    data = request.get_json()
    required_fields = ["LB", "LT", "KT", "KM", "GRS"]
    for field in required_fields:
        if field not in data:
            return jsonify({
                "status": "gagal",
                "error": f"Field '{field}' wajib disertakan."
            }), 400

    try:
        lb = float(data["LB"])
        lt = float(data["LT"])
        kt = int(data["KT"])
        km = int(data["KM"])
        grs = int(data["GRS"])

        if lb <= 0 or lt <= 0 or kt < 0 or km < 0 or grs < 0:
            return jsonify({
                "status": "gagal",
                "error": "Nilai LB dan LT harus > 0, serta KT, KM, GRS harus >= 0."
            }), 400

        df_input = pd.DataFrame([{
            "LB": lb,
            "LT": lt,
            "KT": kt,
            "KM": km,
            "GRS": grs
        }], columns=["LB", "LT", "KT", "KM", "GRS"])

        raw_pred = house_model.predict(df_input)
        estimasi_nilai = float(raw_pred[0])

        return jsonify({
            "estimasi_harga": estimasi_nilai,
            "status": "sukses"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "gagal",
            "error": f"Terjadi kesalahan saat memprediksi harga: {str(e)}"
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint Prediksi Gambar.
    Menerima file gambar via multipart/form-data (key: 'image' atau 'file')
    atau Base64 via JSON body (key: 'image_base64').
    """
    image_stream = None
    predictor = get_image_predictor()

    # 1. Penanganan Multipart Form Data (Upload File langsung dari Frontend)
    if "image" in request.files or "file" in request.files:
        file_obj = request.files.get("image") or request.files.get("file")
        if file_obj.filename == "":
            return jsonify({
                "success": False,
                "error": "Tidak ada file yang dipilih (nama file kosong)."
            }), 400

        if not is_allowed_file(file_obj.filename):
            return jsonify({
                "success": False,
                "error": f"Format file tidak didukung. Format yang diizinkan: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        image_stream = io.BytesIO(file_obj.read())

    # 2. Penanganan JSON Base64 (Alternatif opsional untuk Frontend)
    elif request.is_json and request.json.get("image_base64"):
        try:
            b64_data = request.json.get("image_base64")
            if "," in b64_data:
                # Menghapus prefix data:image/...;base64, jika ada
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            image_stream = io.BytesIO(image_bytes)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Gagal memproses string Base64 gambar: {str(e)}"
            }), 400

    else:
        return jsonify({
            "success": False,
            "error": "Harap kirimkan file gambar melalui form-data dengan key 'image' atau 'file'."
        }), 400

    # Jalankan prediksi
    try:
        result = predictor.predict(image_stream)
        return jsonify({
            "success": True,
            **result
        }), 200
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan saat memproses gambar: {str(e)}"
        }), 500


@app.route("/predict-handwrite", methods=["POST"])
@app.route("/predict-handwrite/", methods=["POST"])
def predict_handwrite():
    """
    Endpoint Prediksi Angka Tulisan Tangan (MNIST).
    Menerima JSON dengan format: {"image_base64": "..."}
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Payload harus berformat JSON dengan header Content-Type: application/json."
        }), 400

    data = request.get_json()
    image_base64 = data.get("image_base64")

    if not image_base64:
        return jsonify({
            "success": False,
            "error": "Field 'image_base64' wajib diisi."
        }), 400

    try:
        predictor = get_mnist_predictor()
        result = predictor.predict(image_base64)
        return jsonify({
            "predicted_digit": result["predicted_digit"],
            "confidence": result["confidence"]
        }), 200
    except ValueError as ve:
        return jsonify({
            "success": False,
            "error": str(ve)
        }), 400
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Terjadi kesalahan saat memproses prediksi angka: {str(e)}"
        }), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Endpoint tidak ditemukan. Cek GET / untuk daftar endpoint yang tersedia."
    }), 404


@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    max_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return jsonify({
        "success": False,
        "error": f"Ukuran file terlalu besar. Maksimum ukuran file adalah {max_mb} MB."
    }), 413


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Terjadi kesalahan internal server."
    }), 500


if __name__ == "__main__":
    print(f"[INFO] Server berjalan di http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)


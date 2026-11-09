import io
import base64
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
from src.predictor import predictor
try:
    from src.mnist_predictor import mnist_predictor
except Exception as e:
    print(f"[WARNING] Tidak dapat memuat mnist_predictor: {e}")
    mnist_predictor = None

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
    Memeriksa apakah server berjalan dan model telah berhasil dimuat.
    """
    model_ready = predictor.is_ready()
    status_code = 200 if model_ready else 503

    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "model_path": MODEL_PATH,
        "target_classes": CLASS_NAMES,
        "input_shape": [150, 150, 3]
    }), status_code


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint Prediksi Gambar.
    Menerima file gambar via multipart/form-data (key: 'image' atau 'file')
    atau Base64 via JSON body (key: 'image_base64').
    """
    if not predictor.is_ready():
        return jsonify({
            "success": False,
            "error": "Model machine learning belum siap atau gagal dimuat."
        }), 503

    image_stream = None

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
    if mnist_predictor is None or not mnist_predictor.is_ready():
        return jsonify({
            "success": False,
            "error": "Model MNIST CNN belum siap atau gagal dimuat."
        }), 503

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
        result = mnist_predictor.predict(image_base64)
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


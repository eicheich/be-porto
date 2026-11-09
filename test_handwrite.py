import io
import base64
import numpy as np
from PIL import Image, ImageDraw

def create_sample_digit_image():
    # Buat kanvas putih ukuran 280x280 (seperti canvas di browser)
    img = Image.new("RGBA", (280, 280), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Gambar garis angka 1 tebal di tengah
    draw.line([(140, 40), (140, 240)], fill=(0, 0, 0, 255), width=24)
    draw.line([(100, 80), (140, 40)], fill=(0, 0, 0, 255), width=24)
    draw.line([(90, 240), (190, 240)], fill=(0, 0, 0, 255), width=24)
    
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def test_fastapi():
    print("\n--- Menguji FastAPI (/predict-handwrite/) ---")
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    
    # 1. Health check
    res_health = client.get("/health")
    print("Health response:", res_health.json())
    assert res_health.status_code == 200

    # 2. Test predict handwrite
    sample_b64 = create_sample_digit_image()
    res = client.post("/predict-handwrite/", json={"image_base64": sample_b64})
    print("FastAPI /predict-handwrite/ response:", res.status_code, res.json())
    assert res.status_code == 200
    data = res.json()
    assert "predicted_digit" in data
    assert "confidence" in data
    assert isinstance(data["predicted_digit"], int)
    assert 0.0 <= data["confidence"] <= 1.0
    print(f"Hasil prediksi FastAPI: Digit {data['predicted_digit']} (Confidence: {data['confidence']})")

def test_flask():
    print("\n--- Menguji Flask (/predict-handwrite/) ---")
    from app import app as flask_app
    client = flask_app.test_client()

    sample_b64 = create_sample_digit_image()
    res = client.post("/predict-handwrite/", json={"image_base64": sample_b64})
    print("Flask /predict-handwrite/ response:", res.status_code, res.get_json())
    assert res.status_code == 200
    data = res.get_json()
    assert "predicted_digit" in data
    assert "confidence" in data
    assert isinstance(data["predicted_digit"], int)
    assert 0.0 <= data["confidence"] <= 1.0
    print(f"Hasil prediksi Flask: Digit {data['predicted_digit']} (Confidence: {data['confidence']})")

if __name__ == "__main__":
    test_fastapi()
    test_flask()
    print("\n=== SEMUA PENGUJIAN PREDIKSI ANGKA (MNIST) BERHASIL! ===")


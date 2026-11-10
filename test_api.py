import io
import time
from PIL import Image
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    print("\n--- [1] Menguji Endpoint GET /health & GET / ---")
    res_root = client.get("/")
    print(f"Status Code /: {res_root.status_code}")
    print("Response JSON /:", res_root.json())
    assert res_root.status_code == 200

    res_health = client.get("/health")
    print(f"Status Code /health: {res_health.status_code}")
    print("Response JSON /health:", res_health.json())
    assert res_health.status_code == 200
    print(">> Health check BERHASIL.")

def test_predict_form_data():
    print("\n--- [2] Menguji Endpoint POST /predict (Multipart Form-Data) ---")
    img = Image.new("RGB", (150, 150), color=(128, 64, 200))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    files = {"image": ("test_image.jpg", img_bytes.getvalue(), "image/jpeg")}
    start = time.time()
    res = client.post("/predict", files=files)
    latency = (time.time() - start) * 1000

    print(f"Status Code: {res.status_code}")
    print(f"Client Latency: {latency:.2f} ms")
    print("Response JSON:", res.json())
    assert res.status_code == 200
    assert res.json().get("success") is True
    print(">> Prediksi gambar BERHASIL.")

def test_validation_errors():
    print("\n--- [3] Menguji Validasi Error ---")
    # Kasus: Request tanpa file
    res = client.post("/predict")
    print(f"Tanpa file -> Status: {res.status_code}, Detail: {res.json()}")
    assert res.status_code == 400

    # Kasus: Format file tidak didukung
    files = {"image": ("document.txt", b"ini bukan gambar", "text/plain")}
    res = client.post("/predict", files=files)
    print(f"File TXT -> Status: {res.status_code}, Detail: {res.json()}")
    assert res.status_code == 400

if __name__ == "__main__":
    print("Memulai simulasi pengujian backend FastAPI TestClient...")
    test_health()
    test_predict_form_data()
    test_validation_errors()
    print("\n=== SEMUA PENGUJIAN API BERHASIL! ===")

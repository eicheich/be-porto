import io
import time
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    print("\n--- [1] Menguji Endpoint GET /health ---")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {res.status_code}")
        print("Response JSON:")
        print(res.json())
        assert res.status_code == 200, "Health check gagal!"
        print(">> Health check BERHASIL.")
    except Exception as e:
        print(f"Error saat health check: {e}")

def test_predict_form_data():
    print("\n--- [2] Menguji Endpoint POST /predict (Multipart Form-Data) ---")
    try:
        # Buat gambar dummy 150x150 berwarna RGB untuk pengujian
        img = Image.new("RGB", (150, 150), color=(128, 64, 200))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"image": ("test_image.jpg", img_bytes, "image/jpeg")}
        start = time.time()
        res = requests.post(f"{BASE_URL}/predict", files=files)
        latency = (time.time() - start) * 1000

        print(f"Status Code: {res.status_code}")
        print(f"Client Latency: {latency:.2f} ms")
        print("Response JSON:")
        print(res.json())
        assert res.status_code == 200, "Prediksi gagal!"
        print(">> Prediksi gambar BERHASIL.")
    except Exception as e:
        print(f"Error saat pengujian prediksi: {e}")

def test_validation_errors():
    print("\n--- [3] Menguji Validasi Error ---")
    # Kasus: Request tanpa file
    res = requests.post(f"{BASE_URL}/predict")
    print(f"Tanpa file -> Status: {res.status_code}, Response: {res.json()}")

    # Kasus: Format file tidak didukung
    dummy_txt = io.BytesIO(b"ini bukan gambar")
    files = {"image": ("document.txt", dummy_txt, "text/plain")}
    res = requests.post(f"{BASE_URL}/predict", files=files)
    print(f"File TXT -> Status: {res.status_code}, Response: {res.json()}")

if __name__ == "__main__":
    print(f"Memulai simulasi pengujian backend di {BASE_URL}...")
    test_health()
    test_predict_form_data()
    test_validation_errors()


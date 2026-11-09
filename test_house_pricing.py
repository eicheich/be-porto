import requests
import time

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("\n--- [1] Menguji Endpoint GET / (Health Check) ---")
    try:
        res = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {res.status_code}")
        print("Response JSON:", res.json())
        assert res.status_code == 200
        print(">> Health Check BERHASIL")
    except Exception as e:
        print(f"Error: {e}")

def test_predict_success():
    print("\n--- [2] Menguji Endpoint POST /predict-house-pricing (Valid Input) ---")
    payload = {
        "LB": 120.0,
        "LT": 150.0,
        "KT": 3,
        "KM": 2,
        "GRS": 1
    }
    try:
        res = requests.post(f"{BASE_URL}/predict-house-pricing", json=payload)
        print(f"Status Code: {res.status_code}")
        print("Response JSON:", res.json())
        assert res.status_code == 200
        assert res.json().get("status") == "sukses"
        print(f">> Estimasi Harga: Rp {res.json().get('estimasi_harga'):,.2f}")
        print(">> Prediksi Harga Rumah BERHASIL")
    except Exception as e:
        print(f"Error: {e}")

def test_predict_validation_error():
    print("\n--- [3] Menguji Validasi Input Pydantic (Invalid Input: LB <= 0) ---")
    payload = {
        "LB": 0,    # Harusnya > 0
        "LT": 150.0,
        "KT": -1,   # Harusnya >= 0
        "KM": 2,
        "GRS": 1
    }
    try:
        res = requests.post(f"{BASE_URL}/predict-house-pricing", json=payload)
        print(f"Status Code: {res.status_code}")
        print("Response JSON:", res.json())
        assert res.status_code == 422  # Unprocessable Entity dari Pydantic
        print(">> Validasi Pydantic BERHASIL menangkap error (Status 422)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"Menguji FastAPI House Pricing di {BASE_URL}...")
    test_health()
    test_predict_success()
    test_predict_validation_error()


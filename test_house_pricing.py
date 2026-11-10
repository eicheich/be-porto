from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    print("\n--- [1] Menguji Endpoint GET / (Health Check) ---")
    res = client.get("/")
    print(f"Status Code: {res.status_code}")
    print("Response JSON:", res.json())
    assert res.status_code == 200
    print(">> Health Check BERHASIL")

def test_predict_success():
    print("\n--- [2] Menguji Endpoint POST /predict-house-pricing (Valid Input) ---")
    payload = {
        "LB": 120.0,
        "LT": 150.0,
        "KT": 3,
        "KM": 2,
        "GRS": 1
    }
    res = client.post("/predict-house-pricing", json=payload)
    print(f"Status Code: {res.status_code}")
    print("Response JSON:", res.json())
    assert res.status_code == 200
    assert res.json().get("status") == "sukses"
    print(f">> Estimasi Harga: Rp {res.json().get('estimasi_harga'):,.2f}")
    print(">> Prediksi Harga Rumah BERHASIL")

def test_predict_validation_error():
    print("\n--- [3] Menguji Validasi Input Pydantic (Invalid Input: LB <= 0) ---")
    payload = {
        "LB": 0,    # Harusnya > 0
        "LT": 150.0,
        "KT": -1,   # Harusnya >= 0
        "KM": 2,
        "GRS": 1
    }
    res = client.post("/predict-house-pricing", json=payload)
    print(f"Status Code: {res.status_code}")
    print("Response JSON:", res.json())
    assert res.status_code == 422  # Unprocessable Entity dari Pydantic
    print(">> Validasi Pydantic BERHASIL menangkap error (Status 422)")

if __name__ == "__main__":
    print("Menguji FastAPI House Pricing via TestClient...")
    test_health()
    test_predict_success()
    test_predict_validation_error()
    print("\n=== SEMUA PENGUJIAN HOUSE PRICING BERHASIL! ===")

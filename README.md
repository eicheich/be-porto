# Backend API Portfolio: House Pricing (FastAPI) & Image Classification (Flask)

Backend API yang menyediakan layanan Machine Learning:
1. **House Pricing Prediction (FastAPI)**: Model estimasi harga rumah berbasis Random Forest Regressor (`model_random_forest.joblib`).
2. **Image Classification (Flask)**: Model klasifikasi gambar 3 kelas (cats, dogs, others) berbasis Keras (`model_classification_3class.keras`).
3. **Indo Text-to-SQL**: Model konversi natural language ke SQL (Qwen2.5-Coder LoRA).

---

## 📁 Struktur Direktori

```text
be-porto/
│
├── main.py                                  # Entrypoint FastAPI (House Pricing Prediction)
├── app.py                                   # Entrypoint Flask (Image Classification & Text-to-SQL)
├── test_house_pricing.py                    # Skrip pengujian FastAPI House Pricing
├── test_api.py                              # Skrip pengujian Flask API
├── requirements.txt                         # Daftar dependensi library Python
│
├── mode/ & model/
│   ├── model_random_forest.joblib           # Model Regresi Harga Rumah
│   └── model_classification_3class.keras    # Model Keras Klasifikasi Gambar
│
└── src/
    ├── config.py
    ├── predictor.py
    └── sql_generator.py
```

---

## 🚀 Menjalankan Server FastAPI (House Pricing Prediction)

Jalankan perintah ini di terminal:
```bash
uvicorn main:app --reload
```
*(Server akan berjalan di `http://127.0.0.1:8000` dengan Swagger UI interaktif di `http://127.0.0.1:8000/docs`)*

---

## 📡 Dokumentasi Endpoint House Pricing (FastAPI)

### 1. Health Check
- **URL**: `GET /`
- **Response (200 OK)**:
  ```json
  {
    "status": "online",
    "message": "House Pricing Prediction API is running",
    "model_loaded": true,
    "features_required": ["LB", "LT", "KT", "KM", "GRS"]
  }
  ```

### 2. Prediksi Harga Rumah
- **URL**: `POST /predict-house-pricing`
- **Request Body (JSON)**:
  ```json
  {
    "LB": 120.0,
    "LT": 150.0,
    "KT": 3,
    "KM": 2,
    "GRS": 1
  }
  ```
- **Response Sukses (200 OK)**:
  ```json
  {
    "estimasi_harga": 2777180000.0,
    "status": "sukses"
  }
  ```
- **Validasi Data**:
  - `LB` (Luas Bangunan): wajib, > 0
  - `LT` (Luas Tanah): wajib, > 0
  - `KT` (Kamar Tidur): wajib, >= 0
  - `KM` (Kamar Mandi): wajib, >= 0
  - `GRS` (Garasi): wajib, >= 0

### 3. Prediksi Angka Tulisan Tangan (MNIST CNN)
- **URL**: `POST /predict-handwrite/` (atau `POST /predict-handwrite`)
- **Headers**: `Content-Type: application/json`
- **Request Body (JSON)**:
  ```json
  {
    "image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
  ```
- **Response Sukses (200 OK)**:
  ```json
  {
    "predicted_digit": 7,
    "confidence": 0.9982
  }
  ```
- **Alur Preprocessing Gambar**:
  1. Membersihkan prefix `data:image/...;base64,`.
  2. Mendecode string base64 menjadi gambar (PIL).
  3. Mengonversi alpha channel transparan menjadi background putih solid.
  4. Mengubah gambar menjadi Grayscale (1 channel).
  5. Melakukan resize menjadi 28x28 pixel.
  6. Menginversi warna secara otomatis jika gambar canvas berlatar belakang putih dan coretan hitam (`ImageOps.invert`).
  7. Normalisasi nilai pixel (dibagi 255.0) dan reshape menjadi `(1, 28, 28, 1)`.
  8. Menjalankan inferensi pada model `model/mnist_cnn_model.keras`.

---

## 🚀 Menjalankan Server Flask (Image Classification)

```bash
python app.py
```
*(Server Flask berjalan di `http://127.0.0.1:5000`)*

---

## 🚀 Panduan Memulai

### 1. Prasyarat
- Python 3.10+ (Direkomendasikan Python 3.11 atau 3.12)
- Pip

### 2. Instalasi Dependensi

```bash
# Opsional: Buat dan aktifkan virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

### 3. Menjalankan Server

Anda dapat menjalankan backend ini dengan salah satu cara berikut:

**Opsi A — Python Flask (Rekomendasi Utama)**:
```bash
python app.py
```

**Opsi B — Uvicorn (ASGI)**:
```bash
uvicorn main:app --reload --port 5000
```

> **Catatan Penting**: Frontend Anda di `porto_web` secara default memanggil endpoint di `http://localhost:5000/predict`. Oleh karena itu, jika menggunakan Uvicorn, **pastikan sertakan flag `--port 5000`** (bukan port default 8000).

---

## 📡 Dokumentasi Endpoint API

### 1. Cek Kesiapan Server (Health Check)
Endpoint untuk memeriksa apakah server aktif dan model ML siap di memori.

- **URL**: `/health`
- **Method**: `GET`
- **Response Sukses (200 OK)**:
  ```json
  {
    "input_shape": [150, 150, 3],
    "model_loaded": true,
    "model_path": ".../model/model_classification_3class.keras",
    "status": "healthy",
    "target_classes": [
      "cats",
      "dogs",
      "others"
    ]
  }
  ```

---

### 2. Prediksi Klasifikasi Gambar
Endpoint untuk mengunggah file gambar dan menerima hasil klasifikasi.

- **URL**: `/predict`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body Parameter**:
  - `image` *(File, Wajib)*: File gambar berekstensi `.jpg`, `.jpeg`, `.png`, atau `.webp`.
  *(Catatan: key `file` juga diterima)*

- **Response Sukses (200 OK)**:
  ```json
  {
    "success": true,
    "prediction": "cats",
    "confidence": 0.9421,
    "probabilities": {
      "cats": 0.9421,
      "dogs": 0.0452,
      "others": 0.0127
    },
    "execution_time_ms": 32.5
  }
  ```

- **Response Gagal (400 / 413 / 500)**:
  ```json
  {
    "success": false,
    "error": "Format file tidak didukung. Format yang diizinkan: png, jpg, jpeg, webp"
  }
  ```

---

## 🧪 Contoh Pengujian

### A. Menggunakan cURL

```bash
# Health check
curl -X GET http://127.0.0.1:5000/health

# Prediksi gambar
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@path/to/kucing.jpg"
```

### B. Menggunakan Postman
1. Buat request baru dengan method **POST**.
2. Masukkan URL: `http://localhost:5000/predict`.
3. Buka tab **Body** $\rightarrow$ pilih **form-data**.
4. Pada kolom **Key**, ketik `image` dan ubah tipe dropdown dari *Text* menjadi *File*.
5. Pada kolom **Value**, pilih file gambar dari komputer Anda.
6. Klik **Send**.

---

## 💻 Panduan Integrasi Frontend (JavaScript / React / Vue)

Karena Anda akan membuat Frontend di project terpisah, fitur **CORS** sudah diaktifkan secara default di backend ini. Berikut contoh kode pemanggilan API menggunakan `fetch` dari frontend:

```javascript
// Contoh fungsi upload gambar di Frontend (React / Vue / Vanilla JS)
async function uploadAndClassifyImage(file) {
  const formData = new FormData();
  formData.append("image", file); // 'image' sesuai key di backend

  try {
    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
      // Jangan set header 'Content-Type' secara manual, browser akan mengisinya otomatis beserta boundary-nya
    });

    const data = await response.json();
    if (data.success) {
      console.log("Prediksi:", data.prediction);
      console.log("Tingkat Keyakinan (Confidence):", (data.confidence * 100).toFixed(2) + "%");
      console.log("Detail Probabilitas:", data.probabilities);
      return data;
    } else {
      console.error("Error dari backend:", data.error);
    }
  } catch (error) {
    console.error("Gagal terhubung ke backend:", error);
  }
}
```


# Portfolio Machine Learning Backend API (FastAPI + Docker)

Backend API production-ready berkinerja tinggi berbasis **FastAPI**, dioptimalkan untuk deployment container pada **Microsoft Azure** (Azure Container Apps / App Service) dengan **biaya paling efisien (Lowest Cost)**.

---

## 🌟 Fitur Utama & Model ML
1. **House Pricing Prediction**: Model Random Forest Regressor (`model/model_random_forest.joblib`).
2. **Handwritten Digit Recognizer**: Model Deep Learning CNN MNIST 10-Class (`model/mnist_cnn_model.keras`).
3. **Cats vs Dogs Image Classifier**: Model CNN Keras 3-Class (`model/model_classification_3class.keras`).
4. **Thread-Safe Lazy Loading**: Model hanya dimuat ke memori saat endpoint pertama kali menerima request (startup container < 0.2 detik).
5. **Non-Blocking Asynchronous Processing**: Seluruh inferensi CPU-bound dijalankan di worker thread pool (`run_in_threadpool`), menjaga event loop tetap responsif.
6. **Security Hardening**: Validasi batas ukuran file (max 5 MB), batas panjang base64 (max 2 MB), sanitasi ekstensi gambar, dan perlindungan Decompression Bomb via Pillow.

---

## 📁 Struktur Direktori

```text
be-porto/
│
├── main.py                     # Entrypoint utama FastAPI & route handlers
├── requirements.txt            # Dependensi lean & hemat memori (tanpa PyTorch/Flask)
├── Dockerfile                  # Container production Linux slim (Non-root user)
├── .dockerignore               # Optimasi layer cache & ukuran image
│
├── model/                      # Direktori penyimpanan bobot model
│   ├── mnist_cnn_model.keras
│   ├── model_classification_3class.keras
│   └── model_random_forest.joblib
│
├── src/
│   ├── config.py               # Konfigurasi path, batas upload, & CORS
│   ├── predictor.py            # Predictor & Lazy Loader Klasifikasi Gambar (Thread-safe)
│   └── mnist_predictor.py      # Predictor & Lazy Loader MNIST Digit (Thread-safe)
│
├── test_api.py                 # Pengujian otomatis endpoint Klasifikasi Gambar
├── test_handwrite.py           # Pengujian otomatis endpoint MNIST Digit
└── test_house_pricing.py       # Pengujian otomatis endpoint Harga Rumah
```

---

## 🚀 Menjalankan Secara Lokal

### 1. Menggunakan Python Virtual Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Akses Swagger UI interaktif di: **`http://localhost:8000/docs`**

### 2. Menggunakan Docker Secara Lokal
```bash
# Build image
docker build -t be-porto:latest .

# Run container
docker run -p 8000:8000 --name ml-api be-porto:latest
```
Uji health check:
```bash
curl http://localhost:8000/health
```

---

## ☁️ Panduan Deploy ke Azure Container Apps (Biaya Terendah / $0)

Azure Container Apps (ACA) pada **Consumption Plan** memberikan kuota gratis setiap bulan:
- **180.000 vCPU-detik** & **360.000 GiB-detik** gratis.
- **2 juta request** per bulan gratis.
- Fitur **Scale-to-Zero** otomatis mematikan container saat tidak ada pengunjung sehingga tagihan **$0**.

### Langkah 1: Push Image ke GitHub Container Registry (Gratis)
```bash
# Login ke GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag dan push image
docker build -t ghcr.io/USERNAME/be-porto:latest .
docker push ghcr.io/USERNAME/be-porto:latest
```

### Langkah 2: Deploy di Azure CLI
```bash
# Buat resource group
az group create --name rg-portfolio --location southeastasia

# Buat Container Apps Environment
az containerapp env create \
  --name env-portfolio \
  --resource-group rg-portfolio \
  --location southeastasia

# Deploy Container App dengan Scale-to-Zero & RAM 1 GB
az containerapp create \
  --name api-ml \
  --resource-group rg-portfolio \
  --environment env-portfolio \
  --image ghcr.io/USERNAME/be-porto:latest \
  --target-port 8000 \
  --ingress external \
  --cpu 0.5 \
  --memory 1.0Gi \
  --min-replicas 0 \
  --max-replicas 2
```

---

## 📡 Daftar Endpoint API

| Method | Endpoint | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Root status & pengecekan ketersediaan file model |
| `GET` | `/health` | Health probe container & monitoring status model di RAM |
| `POST` | `/predict-house-pricing` | Estimasi harga rumah (JSON payload: `LB, LT, KT, KM, GRS`) |
| `POST` | `/predict-handwrite/` | Prediksi angka tulisan tangan (JSON payload: `image_base64`) |
| `POST` | `/predict` | Klasifikasi hewan/gambar (Multipart `image` file atau JSON `image_base64`) |

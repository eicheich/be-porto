# ==============================================================================
# Multi-stage or Optimized Single-stage Dockerfile for Azure Container Apps / App Service
# Designed for Lowest Memory Footprint, Fast Boot, and Security Hardening
# ==============================================================================
FROM python:3.11-slim-bookworm

# Environment Variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    CUDA_VISIBLE_DEVICES="-1" \
    TF_CPP_MIN_LOG_LEVEL="2" \
    TF_NUM_INTRAOP_THREADS="2" \
    TF_NUM_INTEROP_THREADS="1" \
    OMP_NUM_THREADS="2"

WORKDIR /app

# Install minimal OS dependencies for healthcheck & security
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security compliance
RUN useradd -u 10001 -m -s /bin/bash appuser

# Install Python dependencies first (leveraging Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY model/ ./model/
COPY main.py .

# Adjust permissions for non-root execution
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose internal container port
EXPOSE 8000

# Container Healthcheck (Digunakan oleh Azure Container Apps / Docker Engine)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn Single-Worker (Hemat RAM di cloud, scale dilakukan secara horizontal)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ── Base: PyTorch oficial con CUDA 12.1 (compatible con RTX 3xxx/4xxx, A100, H100) ──
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Dependencias de sistema para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir boto3  # para S3-compatible storage (Crusoe/R2/S3)

# Código fuente y modelos
COPY . .

# El modelo yolov8n.pt se descarga automáticamente por ultralytics si no existe.
# tennis_ball_best.pt debe copiarse antes de build o montarse en un volume.

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — Setup inicial de la VM GPU (Crusoe / GCP / cualquier Linux NVIDIA)
#
# Uso:
#   chmod +x deploy.sh
#   ./deploy.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo "=== PaddelStats — Setup servidor GPU ==="

# ── 1. Docker ────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/5] Instalando Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "      Docker instalado. Es posible que necesites volver a iniciar sesión."
else
    echo "[1/5] Docker ya instalado."
fi

# ── 2. NVIDIA Container Toolkit ──────────────────────────────────────────────
if ! dpkg -l | grep -q nvidia-container-toolkit 2>/dev/null; then
    echo "[2/5] Instalando NVIDIA Container Toolkit..."
    distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L "https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list" | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "      NVIDIA Container Toolkit instalado."
else
    echo "[2/5] NVIDIA Container Toolkit ya instalado."
fi

# ── 3. Variables de entorno ───────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "[3/5] Creando .env desde .env.example..."
    cp .env.example .env
    echo "      ⚠️  Edita .env antes de continuar: nano .env"
    echo "      (configura S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY, etc.)"
    exit 0
else
    echo "[3/5] .env ya existe."
fi

# ── 4. Build y arranque ───────────────────────────────────────────────────────
echo "[4/5] Construyendo imágenes Docker..."
docker compose build

echo "[5/5] Arrancando servicios..."
docker compose up -d

echo ""
echo "=== ¡Listo! ==="
echo "Backend API:  http://$(curl -s ifconfig.me):8000"
echo "Logs:         docker compose logs -f backend"
echo "Parar:        docker compose down"

"""
Capa de abstracción de almacenamiento.

Soporta dos backends según la variable de entorno STORAGE_BACKEND:
  - "local"  → ficheros en el sistema local (desarrollo, staging)
  - "s3"     → S3-compatible (Crusoe Object Storage, Cloudflare R2, AWS S3)

Uso:
    from .storage import storage
    path = await storage.save(file_bytes, "video.mov")
    local_path = await storage.download_tmp(path)   # descarga a /tmp para procesar
    url = storage.public_url(path)                  # URL firmada de acceso
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

BACKEND = os.environ.get("STORAGE_BACKEND", "local").lower()
LOCAL_DIR = Path(os.environ.get("LOCAL_UPLOAD_DIR", "uploaded_videos"))


class LocalStorage:
    """Almacenamiento en disco local — válido para dev y VMs con volumen persistente."""

    def __init__(self):
        LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, src_path: str, filename: str) -> str:
        """
        Mueve/copia src_path al directorio local.
        Devuelve la ruta relativa al fichero guardado.
        """
        dest = LOCAL_DIR / filename
        shutil.copy2(src_path, dest)
        return str(dest)

    def download_tmp(self, stored_path: str) -> str:
        """El fichero ya es local — devuelve la ruta tal cual."""
        return stored_path

    def cleanup_tmp(self, tmp_path: str, stored_path: str):
        """No hay nada que limpiar en local."""
        pass

    def public_url(self, stored_path: str) -> str:
        """Devuelve una URL relativa para servir el fichero via API."""
        return f"/files/{Path(stored_path).name}"


class S3Storage:
    """Almacenamiento en S3-compatible (Crusoe, AWS, Cloudflare R2)."""

    def __init__(self):
        import boto3
        self.bucket = os.environ["S3_BUCKET"]
        self.client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ["S3_ACCESS_KEY"],
            aws_secret_access_key=os.environ["S3_SECRET_KEY"],
            region_name=os.environ.get("S3_REGION", "auto"),
        )
        print(f"[Storage] S3 backend — bucket: {self.bucket}")

    def save(self, src_path: str, filename: str) -> str:
        """Sube el fichero al bucket. Devuelve la key S3."""
        key = f"uploads/{filename}"
        self.client.upload_file(src_path, self.bucket, key)
        return key

    def download_tmp(self, key: str) -> str:
        """Descarga el fichero a /tmp y devuelve la ruta local temporal."""
        suffix = Path(key).suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        self.client.download_fileobj(self.bucket, key, tmp)
        tmp.close()
        return tmp.name

    def cleanup_tmp(self, tmp_path: str, stored_path: str):
        """Elimina el fichero temporal y sube el resultado si existe."""
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    def upload_result(self, local_path: str, video_id: str) -> str:
        """Sube el vídeo anotado al bucket."""
        key = f"results/{video_id}_annotated.mp4"
        self.client.upload_file(local_path, self.bucket, key)
        return key

    def public_url(self, key: str, expires: int = 3600) -> str:
        """Genera una URL firmada de acceso temporal."""
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )


def _create_storage():
    if BACKEND == "s3":
        return S3Storage()
    return LocalStorage()


# Instancia global
storage = _create_storage()

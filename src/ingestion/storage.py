import os
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.config import load_config

logger = get_logger(__name__)
cfg = load_config()

def save_local(bytes_data: bytes, dest_path: str) -> str:
    try:
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(bytes_data)
        tmp.replace(dest)
        logger.info(f"Saved local file {dest}")
        return str(dest)
    except Exception as e:
        raise CustomException("Failed to save local file", e)

# Optional: MinIO uploader (S3-compatible)
try:
    from minio import Minio
except Exception:
    Minio = None

def upload_to_minio(local_path: str, remote_key: str) -> bool:
    if not cfg.get("storage", {}).get("use_minio", False):
        raise CustomException("MinIO not enabled in config", Exception("use_minio=false"))
    if Minio is None:
        raise CustomException("minio package not installed", Exception("pip install minio"))

    try:
        mcfg = cfg["storage"]["minio"]
        client = Minio(mcfg["endpoint"], access_key=mcfg["access_key"], secret_key=mcfg["secret_key"], secure=False)
        bucket = mcfg["bucket"]
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
        client.fput_object(bucket, remote_key, local_path)
        logger.info(f"Uploaded {local_path} to minio://{bucket}/{remote_key}")
        return True
    except Exception as e:
        raise CustomException("Failed to upload to MinIO", e)

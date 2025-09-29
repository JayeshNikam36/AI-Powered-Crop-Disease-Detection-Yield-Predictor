import csv
import json
from pathlib import Path
import hashlib
import cv2
from typing import Iterable
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.config import load_config

logger = get_logger(__name__)
cfg = load_config()

SUPPORTED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _file_hash(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _is_image_readable(path: Path) -> bool:
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False

def build_manifest(dataset_dir: str, out_manifest: str = None) -> str:
    """
    Walk dataset_dir; expects classification layout:
       dataset_dir/class_name/*.jpg
    Produce CSV with columns: file_path,label,sha1
    Returns path to manifest file.
    """
    try:
        dataset_dir = Path(dataset_dir)
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
        out_manifest = Path(out_manifest or Path(cfg["data"]["manifests_dir"]) / f"{dataset_dir.name}_manifest.csv")
        out_manifest.parent.mkdir(parents=True, exist_ok=True)

        with open(out_manifest, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file_path", "label", "sha1"])
            for class_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
                label = class_dir.name
                for img_path in class_dir.rglob("*"):
                    if img_path.suffix.lower() not in SUPPORTED_EXT:
                        continue
                    sha1 = _file_hash(img_path)
                    writer.writerow([str(img_path.resolve()), label, sha1])
        logger.info(f"Manifest written to {out_manifest}")
        return str(out_manifest)
    except Exception as e:
        raise CustomException("Failed to build manifest", e)

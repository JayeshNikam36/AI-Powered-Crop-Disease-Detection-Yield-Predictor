import os
import shutil
import zipfile
import tarfile
import hashlib
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.config import load_config

logger = get_logger(__name__)
cfg = load_config()

def _sha256_of_file(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def download_file(url: str, dest: Path, chunk_size: int = 1024*1024) -> Path:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logger.info(f"Downloaded {url} -> {dest}")
        return dest
    except Exception as e:
        raise CustomException("Failed to download file", e)

def extract_archive(archive_path: Path, out_dir: Path) -> Path:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(out_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(out_dir)
        else:
            # not an archive: maybe it's already a folder - copy
            logger.info("Not an archive; attempting to copy")
            if archive_path.is_dir():
                shutil.copytree(archive_path, out_dir, dirs_exist_ok=True)
            else:
                raise CustomException("Unsupported archive format", Exception(str(archive_path)))
        logger.info(f"Extracted archive {archive_path} -> {out_dir}")
        return out_dir
    except Exception as e:
        raise CustomException("Failed to extract archive", e)

def download_and_prepare_dataset(name: str, url: str, out_base: Optional[str] = None, checksum: Optional[str] = None) -> Path:
    """
    Downloads dataset archive from `url`, optionally verifies checksum, and extracts to out_base/name/
    Returns path to extracted folder.
    """
    try:
        out_base = Path(out_base or cfg["data"]["raw_data_dir"])
        out_base.mkdir(parents=True, exist_ok=True)
        archive_path = out_base / f"{name}.archive"
        # Download
        logger.info(f"Starting download of dataset {name} from {url}")
        downloaded = download_file(url, archive_path)
        # Checksum if provided
        if checksum:
            got = _sha256_of_file(downloaded)
            if got != checksum:
                raise CustomException(f"Checksum mismatch for {downloaded}", Exception(f"expected={checksum} got={got}"))
        # Extract
        extracted_dir = out_base / name
        extract_archive(downloaded, extracted_dir)
        return extracted_dir
    except Exception as e:
        raise CustomException("Failed to download and prepare dataset", e)

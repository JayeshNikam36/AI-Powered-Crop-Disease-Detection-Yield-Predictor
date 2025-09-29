from fastapi import APIRouter, UploadFile, File, Form
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.ingestion.storage import save_local
from src.utils.config import load_config
from pathlib import Path

router = APIRouter()
logger = get_logger(__name__)
cfg = load_config()

@router.post("/upload_frame")
async def upload_frame(file: UploadFile = File(...), field_id: str = Form(...)):
    try:
        raw_dir = Path(cfg["data"]["raw_data_dir"]) / "uploads" / field_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / file.filename
        data = await file.read()
        saved = save_local(data, str(dest))
        return {"status": "ok", "path": saved}
    except Exception as e:
        raise CustomException("Failed to handle upload", e)

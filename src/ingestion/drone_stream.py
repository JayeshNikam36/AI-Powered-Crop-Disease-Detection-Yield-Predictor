import os
import time
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone
import cv2
from src.utils.logger import get_logger
from src.utils.exception import CustomException
from src.utils.config import load_config

logger = get_logger(__name__)
cfg = load_config()

def _atomic_write(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)

def capture_from_source(source: str = None, out_dir: str = None, fps: int = None, max_frames: int = None):
    """
    Capture frames from a video file or RTSP stream.
    - source: path to .mp4 or RTSP URL. If None, load from config.
    - out_dir: base output directory. Defaults to config data/raw_data_dir + '/drone'
    - fps: frames per second to sample (defaults to cfg drone.frame_rate)
    - max_frames: stop after saving this many frames (None for unlimited)
    """
    try:
        source = source or cfg.get("drone", {}).get("source")
        fps = fps or cfg.get("drone", {}).get("frame_rate", 1)
        out_dir = Path(out_dir or Path(cfg["data"]["raw_data_dir"]) / "drone")
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise CustomException("Failed to open video source", Exception(f"{source}"))

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        sample_interval = max(1, int(round(video_fps / float(fps))))
        logger.info(f"Video FPS={video_fps}; sampling every {sample_interval} frames to achieve ~{fps} fps")

        saved = 0
        frame_idx = 0
        date_dir = out_dir / datetime.now(timezone.utc).strftime("%Y%m%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        meta_file = date_dir / "metadata.jsonl"

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("Stream ended or frame not available")
                break
            if frame_idx % sample_interval == 0:
                # Save frame
                frame_id = uuid.uuid4().hex
                fname = f"frame_{frame_id}.jpg"
                fpath = date_dir / fname
                # write atomically via imencode then _atomic_write
                success, encimg = cv2.imencode(".jpg", frame)
                if not success:
                    logger.warning(f"Failed to encode frame {frame_idx}")
                else:
                    _atomic_write(fpath, encimg.tobytes())
                    ts = datetime.now(timezone.utc).isoformat()
                    meta = {
                        "frame_id": frame_id,
                        "file_path": str(fpath.resolve()),
                        "timestamp": ts,
                        "source": source,
                        "camera_id": cfg.get("drone", {}).get("camera_id", "camera_0"),
                    }
                    # append metadata JSON line
                    with open(meta_file, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(meta) + "\n")
                    saved += 1
                    logger.info(f"Saved frame {saved}: {fpath}")
                    if max_frames and saved >= max_frames:
                        logger.info("Reached max_frames; stopping capture")
                        break
            frame_idx += 1
        cap.release()
        return str(date_dir)
    except Exception as e:
        raise CustomException("Failed to capture from video source", e)

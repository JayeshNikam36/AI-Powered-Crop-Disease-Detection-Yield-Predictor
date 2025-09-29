# src/scripts/run_ingestion.py
from src.ingestion.dataset_downloader import download_and_prepare_dataset
from src.ingestion.manifest_builder import build_manifest
from src.ingestion.drone_stream import capture_from_source
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    logger.info("Starting ingestion tests...")
    print("Dataset downloader loaded successfully")
    manifest_path = build_manifest("data/raw/fake_dataset")
    print("Manifest created at:", manifest_path)
    out_dir = capture_from_source(source="data/sample/test_video.mp4", fps=1, max_frames=2)
    print("Frames saved to:", out_dir)

if __name__ == "__main__":
    main()

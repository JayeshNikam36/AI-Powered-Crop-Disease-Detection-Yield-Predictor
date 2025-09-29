import os
import shutil
import random
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def split_dataset(
    raw_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Split dataset from raw_dir into train, val, test folders inside output_dir.
    Maintains subfolders (classes) automatically.
    """
    random.seed(seed)
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"{raw_dir} does not exist")

    # Create train, val, test folders
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    # Iterate over classes
    for class_folder in raw_dir.iterdir():
        if not class_folder.is_dir():
            continue

        images = list(class_folder.glob("*"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train+n_val],
            "test": images[n_train+n_val:]
        }

        for split_name, split_images in splits.items():
            split_class_dir = output_dir / split_name / class_folder.name
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                shutil.copy(img_path, split_class_dir)

        logger.info(f"Class {class_folder.name}: {n_train} train, {n_val} val, {n_test} test images")

    logger.info(f"Dataset split complete. Output directory: {output_dir}")

if __name__ == "__main__":
    raw_folder = r"data/raw/PlantVillage"  # your downloaded dataset
    output_folder = r"data/processed"      # folder to save train/val/test
    split_dataset(raw_folder, output_folder)

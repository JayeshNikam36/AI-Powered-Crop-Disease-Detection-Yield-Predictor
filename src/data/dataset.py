import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from src.utils.exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CropDiseaseDataset(Dataset):
    def __init__(self, manifest_file, transform=None):
        """
        Args:
            manifest_file (str or Path): path to CSV with file_path and label
            transform (torchvision.transforms.Compose): preprocessing transforms
        """
        try:
            self.data = pd.read_csv(manifest_file)
            self.transform = transform
        except Exception as e:
            raise CustomException("Failed to initialize dataset", e)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            img_path = row['file_path']
            label = row['label']

            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            raise CustomException(f"Failed to load image at index {idx}", e)

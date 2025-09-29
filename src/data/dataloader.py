from torch.utils.data import DataLoader, random_split
from src.data.dataset import CropDiseaseDataset
from src.data.transforms import get_transforms
from src.utils.config import cfg

def create_dataloaders(manifest_file):
    # Load dataset
    dataset = CropDiseaseDataset(manifest_file, transform=get_transforms(train=True))

    # Split
    train_size = int(len(dataset) * cfg['data']['train_split'])
    val_size = int(len(dataset) * cfg['data']['val_split'])
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['training']['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(data_dir, batch_size=32, img_size=224):
    """
    Create PyTorch dataloaders for train, val, test
    Args:
        data_dir (str): path to processed folder containing train/val/test
        batch_size (int)
        img_size (int)
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

from src.data.dataloader import create_dataloaders
from src.utils.config import cfg

def main():
    manifest_file = "data/manifests/fake_dataset_manifest.csv"
    
    # unpack the tuple
    train_loader, val_loader, test_loader = create_dataloaders(manifest_file=manifest_file)
    
    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    print("Number of test batches:", len(test_loader))
    
    for images, labels in train_loader:
        print("Batch images shape:", images.shape)
        print("Batch labels:", labels)
        break

if __name__ == "__main__":
    main()

from src.data.dataloader import create_dataloaders
from src.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    try:
        manifest_file = "data/manifests/fake_dataset_manifest.csv"
        train_loader, val_loader, test_loader = create_dataloaders(manifest_file)

        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of validation batches: {len(val_loader)}")
        logger.info(f"Number of test batches: {len(test_loader)}")

        # Inspect a single batch
        for images, labels in train_loader:
            logger.info(f"Batch images shape: {images.shape}")
            logger.info(f"Batch labels: {labels}")
            break  # just check the first batch

        print("DataLoader test completed successfully!")

    except Exception as e:
        logger.exception("Failed to test DataLoader")
        raise e

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.config import cfg
from src.datasets.leaf_dataset import create_dataloaders
from src.models.classifier import initialize_classifier
import logging
from tqdm import tqdm  # progress bar

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Device
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataloaders
    data_dir = "data/processed"  # path to processed dataset (train/val/test)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=cfg["training"]["batch_size"]
    )
    num_classes = len(class_names)
    logger.info(f"Detected {num_classes} classes: {class_names}")

    # Initialize model
    model, _ = initialize_classifier(num_classes=num_classes)
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # Training loop
    epochs = min(cfg["training"]["epochs"], 5)  # limit to 5 epochs max for now
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar for batches
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total if total > 0 else 0
            progress_bar.set_postfix(loss=running_loss/len(train_loader), acc=acc)

        # Validation after each epoch
        if val_loader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            val_acc = val_correct / val_total if val_total > 0 else 0
            logger.info(f"Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}")
            model.train()

    logger.info("Training completed!")

    # Final evaluation on test set
    if test_loader is not None:
        logger.info("Evaluating on test set...")
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        test_acc = test_correct / test_total if test_total > 0 else 0
        logger.info(f"Test Accuracy: {test_acc:.4f}")

    # Save the trained model (remove old checkpoint if it exists)
    save_path = "models/checkpoints/classifier.pth"
    if os.path.exists(save_path):
        os.remove(save_path)
        logger.info(f"Removed old checkpoint: {save_path}")

    torch.save(model.state_dict(), save_path)
    logger.info(f"New trained model saved at {save_path}")

if __name__ == "__main__":
    main()

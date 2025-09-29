import torch
import torch.nn as nn
from torch.optim import Adam
from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)

def train_model(model, dataloaders, device=None, epochs=None, learning_rate=None):
    """
    Trains the classifier on the dataset.
    """
    try:
        device = device or torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
        epochs = epochs or cfg['training']['epochs']
        learning_rate = learning_rate or cfg['training']['learning_rate']

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)

        model.to(device)

        for epoch in range(epochs):
            logger.info(f"Epoch [{epoch+1}/{epochs}]")
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in dataloaders['train']:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total
            logger.info(f"Training Loss: {running_loss/len(dataloaders['train']):.4f}, Accuracy: {train_acc:.4f}")

        return model

    except Exception as e:
        logger.exception("Training failed")
        raise e

import torch
import torch.nn as nn
from torchvision import models
from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.config import cfg

logger = get_logger(__name__)

def initialize_classifier(num_classes=None, pretrain=True):
    """
    Initializes a ResNet50 classifier for crop disease detection.
    Args:
        num_classes (int): Number of output classes. Default: 2 for fake dataset
        pretrained (bool): Use pretrained ImageNet weights
    Returns:
        model (nn.Module), device (torch.device)
    """
    try:
        device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        model = models.resnet50(pretrained=pretrain)
        
        if num_classes is None:
            num_classes = 2

        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)
        logger.info(f"ResNet50 classifier initialized with {num_classes} classes")
        return model, device

    except Exception as e:
        logger.exception("Failed to initialize classifier")
        raise e
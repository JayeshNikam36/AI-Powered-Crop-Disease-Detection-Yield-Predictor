import torch
from torchvision import transforms
from PIL import Image
from src.models.classifier import initialize_classifier
from src.utils.config import cfg
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path):
    """Load and transform image for inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dimension
    return img

def main(image_path):
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model (same num_classes as training)
    model, device = initialize_classifier(num_classes=2)
    model.to(device)
    model.eval()

    # Load saved model weights if you saved them after training
    model.load_state_dict(torch.load("models/checkpoints/classifier.pth", map_location=device))

    # Load image
    img = load_image(image_path).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        label = "classA" if predicted.item() == 0 else "classB"

    logger.info(f"Predicted class: {label}")
    return label

if __name__ == "__main__":
    test_image = "data/sample/test_crop.jpg"  # replace with your new image
    main(test_image)

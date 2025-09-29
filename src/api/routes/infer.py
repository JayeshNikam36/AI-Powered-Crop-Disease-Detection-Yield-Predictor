from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
from src.models.classifier import initialize_classifier
from src.utils.config import cfg

router = APIRouter()

# Initialize model once when API starts
device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
model, _ = initialize_classifier(num_classes=15)  # make sure num_classes matches your trained model
model.load_state_dict(torch.load("models/checkpoints/classifier.pth", map_location=device))
model.to(device)
model.eval()

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load image
    image = Image.open(file.file).convert("RGB")
    transform = cfg["inference"]["transform"]  # or define the same transforms as training
    image = transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
    
    # Map index to class
    class_names = cfg["dataset"]["class_names"]  # make sure you saved this in config.yaml
    predicted_class = class_names[pred_idx.item()]
    
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4)
    }

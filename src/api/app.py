# src/api/app.py
import io
import json
import base64
import logging
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms

# model initializer from your repo
from src.models.classifier import initialize_classifier

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Configuration
# -----------------------
MODEL_CHECKPOINT = Path("models/checkpoints/classifier.pth")
CLASS_NAMES_PATH = Path("models/checkpoints/class_names.json")
# prefer src/knowledge_base but fall back to models/checkpoints
KNOWLEDGE_BASE_CANDIDATES = [
    Path("src/knowledge_base/knowledge_base.json"),
    Path("models/checkpoints/knowledge_base.json"),
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# -----------------------
# Helpers
# -----------------------
def load_class_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, list):
        return data
    # fallback if saved as dict index->name
    if isinstance(data, dict):
        try:
            return [v for k, v in sorted(data.items(), key=lambda x: int(x[0]))]
        except Exception:
            return list(data.values())
    raise ValueError("Unknown class_names.json format")

def load_knowledge_base(candidates: List[Path]) -> Dict[str, Any]:
    for p in candidates:
        if p.exists():
            try:
                kb = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(kb, dict):
                    logger.info(f"Loaded knowledge base from {p}")
                    return kb
                logger.warning(f"Knowledge base at {p} exists but is not a dict. Ignoring.")
            except Exception as e:
                logger.warning(f"Could not parse knowledge base {p}: {e}")
    logger.info("No knowledge base found; continuing with empty descriptions.")
    return {}

def pil_to_base64(img_pil: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Grad-CAM helpers (kept as before)
def find_target_module(model: torch.nn.Module):
    try:
        return model.layer4[-1].conv3
    except Exception:
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d found in model")
        return last_conv

def generate_gradcam(model: torch.nn.Module, device: torch.device, input_tensor: torch.Tensor, target_index: int = None, target_module=None):
    model.zero_grad()
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    fh = target_module.register_forward_hook(forward_hook)
    if hasattr(target_module, "register_full_backward_hook"):
        bh = target_module.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    else:
        bh = target_module.register_backward_hook(backward_hook)

    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)  # (1, num_classes)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_prob, pred_class = torch.max(probs, dim=1)
    pred_idx = int(pred_class.item())
    if target_index is None:
        target_index = pred_idx

    # backward for the target class score
    model.zero_grad()
    outputs[0, target_index].backward(retain_graph=True)

    if 'value' not in activations or 'value' not in gradients:
        fh.remove()
        bh.remove()
        raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients")

    act = activations['value']  # (1, C, H, W)
    grad = gradients['value']   # (1, C, H, W)

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)         # (1, C, 1, 1)
    weighted = (weights * act).sum(dim=1, keepdim=True)          # (1, 1, H, W)
    cam = torch.relu(weighted).squeeze().cpu().numpy()           # (H, W)

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    fh.remove()
    bh.remove()
    return cam, pred_idx, float(pred_prob.item())

def overlay_cam_on_pil(original_pil: Image.Image, cam: np.ndarray, alpha=0.5):
    orig = np.array(original_pil.convert("RGB"))
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # BGR
    overlay = cv2.addWeighted(heatmap_color, alpha, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb), Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))

# -----------------------
# Startup: load model / names / kb
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if not MODEL_CHECKPOINT.exists():
    raise FileNotFoundError(f"Model checkpoint not found at {MODEL_CHECKPOINT}. Run training first.")

class_names = load_class_names(CLASS_NAMES_PATH)
knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_CANDIDATES)

model, _ = initialize_classifier(num_classes=len(class_names))
model.load_state_dict(torch.load(str(MODEL_CHECKPOINT), map_location=device))
model.to(device)
model.eval()

target_module = find_target_module(model)

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="Crop Disease Classifier - Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    # read file
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        logger.exception("Failed to read uploaded image")
        raise HTTPException(status_code=400, detail=f"Could not read image file: {e}")

    input_t = preprocess(img).unsqueeze(0)

    try:
        cam, pred_idx, score = generate_gradcam(model, device, input_t, target_index=None, target_module=target_module)
    except Exception as e:
        logger.exception("Grad-CAM failure")
        raise HTTPException(status_code=500, detail=f"Grad-CAM failure: {e}")

    pred_label = class_names[pred_idx]

    overlay_pil, heatmap_pil = overlay_cam_on_pil(img, cam, alpha=0.5)
    overlay_b64 = pil_to_base64(overlay_pil, fmt="PNG")
    heatmap_b64 = pil_to_base64(heatmap_pil, fmt="PNG")
    original_b64 = pil_to_base64(img, fmt="PNG")

    # probabilities
    try:
        with torch.no_grad():
            outputs = model(input_t.to(device))
            probs_tensor = torch.nn.functional.softmax(outputs, dim=1)[0].cpu()
            probs_list = [float(x) for x in probs_tensor.tolist()]
    except Exception:
        probs_list = []

    probabilities_by_class = [
        {"class": cname, "probability": (probs_list[i] if i < len(probs_list) else 0.0)}
        for i, cname in enumerate(class_names)
    ]

    # lookup description
    description = "No description available."
    try:
        entry = knowledge_base.get(pred_label) or knowledge_base.get(str(pred_idx))
        if isinstance(entry, str):
            description = entry
        elif isinstance(entry, dict):
            description = entry.get("description") or entry.get("desc") or entry.get("text") or description
    except Exception:
        pass

    response = {
        "predicted_class": str(pred_label),
        "predicted_index": int(pred_idx),
        "score": float(score),
        "description": description,
        "probabilities": probs_list,
        "probabilities_by_class": probabilities_by_class,
        "class_names": class_names,
        "original_image_base64": original_b64,
        "gradcam_overlay_base64": overlay_b64,
        "gradcam_heatmap_base64": heatmap_b64,
    }

    return JSONResponse(response)
@app.get("/healthz")
async def health():
    return {"status": "ok"}


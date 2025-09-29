# src/training/grad_cam.py
import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.classifier import initialize_classifier
from src.utils.config import cfg

# -------------------------
# Helpers & preprocessing
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def load_class_names(path="models/checkpoints/class_names.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"class_names.json not found at {path}")
    with open(p, "r") as f:
        data = json.load(f)
    # data maybe like {"class_names": [...]} or just list; handle both
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, list):
        return data
    # maybe saved as { "0": "classA", ... }
    if isinstance(data, dict):
        # sort by keys if numeric
        try:
            items = sorted(data.items(), key=lambda x: int(x[0]))
            return [v for _, v in items]
        except Exception:
            return list(data.values())
    raise ValueError("Unknown format for class_names.json")

# -------------------------
# Grad-CAM implementation
# -------------------------
def find_target_module(model):
    """
    Default: choose last Bottleneck conv of resnet (layer4[-1].conv3)
    If model has attribute 'layer4', try that; otherwise fallback to last Conv2d found.
    """
    # try resnet style
    try:
        layer4 = model.layer4
        target = layer4[-1].conv3  # Bottleneck conv3
        return target
    except Exception:
        # fallback: pick last Conv2d module
        last_conv = None
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No Conv2d layer found in model")
        return last_conv

def generate_gradcam(model, device, input_tensor, target_index=None, target_module=None):
    """
    input_tensor: torch.Tensor shape (1,3,H,W) normalized
    target_index: int or None (if None, use predicted class)
    target_module: nn.Module where to hook (last conv)
    returns: heatmap (H,W) normalized 0..1 (numpy)
    """
    model.zero_grad()
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations['value'] = out  # raw activations (batch, C, H, W)

    def backward_hook(module, grad_in, grad_out):
        # grad_out is a tuple
        gradients['value'] = grad_out[0]

    # register hooks
    fh = target_module.register_forward_hook(forward_hook)
    # use register_full_backward_hook if available for modern PyTorch
    if hasattr(target_module, "register_full_backward_hook"):
        bh = target_module.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
    else:
        bh = target_module.register_backward_hook(backward_hook)

    # forward
    input_tensor = input_tensor.to(device)
    outputs = model(input_tensor)  # (1, num_classes)
    if target_index is None:
        pred_prob, pred_class = torch.max(torch.softmax(outputs, dim=1), dim=1)
        target_index = int(pred_class.item())

    # backward: compute gradients for the target class score
    model.zero_grad()
    score = outputs[0, target_index]
    score.backward(retain_graph=True)

    # get activations and gradients
    if 'value' not in activations or 'value' not in gradients:
        # cleanup hooks before raising
        fh.remove()
        bh.remove()
        raise RuntimeError("Could not obtain activations/gradients from hooks.")

    act = activations['value'].detach()       # (1, C, H, W)
    grad = gradients['value'].detach()        # (1, C, H, W)

    # global-average-pool the gradients over spatial dims -> weights
    weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    # weighted sum of activations
    weighted_acts = (weights * act).sum(dim=1, keepdim=True)  # (1,1,H,W)
    cam = torch.relu(weighted_acts.squeeze(0).squeeze(0)).cpu().numpy()  # (H,W)

    # normalize cam to [0,1]
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    # cleanup hooks
    fh.remove()
    bh.remove()
    return cam, target_index

# -------------------------
# Visualization helpers
# -------------------------
def overlay_cam_on_image(original_img_pil, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    original_img_pil: PIL.Image (RGB)
    cam: numpy HxW with values 0..1
    returns: overlay_bgr (numpy uint8 BGR) and heatmap
    """
    orig = np.array(original_img_pil.convert("RGB"))
    h, w = orig.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)  # BGR

    overlay = cv2.addWeighted(heatmap_color, alpha, cv2.cvtColor(orig, cv2.COLOR_RGB2BGR), 1 - alpha, 0)
    return overlay, heatmap_color

# -------------------------
# Main runner
# -------------------------
def run_on_images(image_paths, output_dir, top_k=1, device=None):
    device = device or torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")

    # load class names and model
    class_names = load_class_names("models/checkpoints/class_names.json")
    model, _ = initialize_classifier(num_classes=len(class_names))
    ckpt = Path("models/checkpoints/classifier.pth")
    if not ckpt.exists():
        raise FileNotFoundError("Checkpoint not found. Train model first.")
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.to(device).eval()

    target_module = find_target_module(model)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(image_paths, desc="Grad-CAM images"):
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"Skipping missing: {img_path}")
            continue

        # load original image (for overlay) and preprocessed tensor
        orig_pil = Image.open(str(img_path)).convert("RGB")
        input_t = preprocess(orig_pil).unsqueeze(0)  # 1,C,H,W

        # generate cam
        cam, target_idx = generate_gradcam(model, device, input_t, target_index=None, target_module=target_module)

        # overlay
        overlay, heatmap_color = overlay_cam_on_image(orig_pil, cam, alpha=0.5)

        # prediction label
        with torch.no_grad():
            out = model(input_t.to(device))
            pred = int(torch.argmax(out, dim=1).cpu().item())
        pred_label = class_names[pred]

        # save files
        stem = img_path.stem
        cls_folder = out_dir / pred_label
        cls_folder.mkdir(parents=True, exist_ok=True)
        cam_path = cls_folder / f"{stem}_cam.png"
        overlay_path = cls_folder / f"{stem}_overlay.png"
        heatmap_path = cls_folder / f"{stem}_heatmap.png"

        cv2.imwrite(str(cam_path), (cam * 255).astype("uint8"))  # grayscale cam
        cv2.imwrite(str(heatmap_path), heatmap_color)            # color heatmap
        cv2.imwrite(str(overlay_path), overlay)                 # overlay BGR

    print(f"Grad-CAM outputs saved to: {out_dir}")

# -------------------------
# CLI
# -------------------------
def collect_test_images(test_dir, max_images=5):
    test_dir = Path(test_dir)
    images = list(test_dir.rglob("*.jpg")) + list(test_dir.rglob("*.jpeg")) + list(test_dir.rglob("*.png"))
    return images[:max_images]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", type=str, default="data/processed/test", help="root test folder (contains class subfolders)")
    p.add_argument("--out_dir", type=str, default="models/checkpoints/gradcam", help="where to save gradcam outputs")
    p.add_argument("--max_images", type=int, default=6, help="max number of test images to process")
    args = p.parse_args()

    imgs = collect_test_images(args.test_dir, max_images=args.max_images)
    if len(imgs) == 0:
        print("No test images found under", args.test_dir)
    else:
        run_on_images(imgs, args.out_dir, device=torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu"))

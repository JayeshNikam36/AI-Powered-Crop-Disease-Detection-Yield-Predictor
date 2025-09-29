# src/scripts/evaluate_model.py
import os
import json
import torch
import logging
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt

from src.datasets.leaf_dataset import create_dataloaders
from src.models.classifier import initialize_classifier
from src.utils.config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

def evaluate():
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataloaders (ensure this returns: train, val, test, class_names)
    data_dir = "data/processed"  # adapt if your processed data path differs
    batch_size = cfg["training"].get("batch_size", 32)
    train_loader, val_loader, test_loader, class_names = create_dataloaders(data_dir=data_dir,
                                                                            batch_size=batch_size)
    num_classes = len(class_names)
    logger.info(f"Detected {num_classes} classes for evaluation: {class_names}")

    # Initialize model and load weights
    model, _ = initialize_classifier(num_classes=num_classes)
    model.to(device)
    ckpt_path = Path("models/checkpoints/classifier.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run training to create it.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Run inference on test set
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = float(accuracy_score(all_labels, all_preds))
    prec = float(precision_score(all_labels, all_preds, average="weighted", zero_division=0))
    rec = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    cls_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    # Save outputs
    out_dir = Path("models/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
    }
    save_json(metrics, out_dir / "metrics_summary.json")
    save_json(cls_report, out_dir / "classification_report.json")
    save_json({"class_names": class_names}, out_dir / "class_names.json")

    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    logger.info(f"Evaluation metrics saved to {out_dir}")
    logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    evaluate()

# src/inference/export_onnx.py
import json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort

from src.models.classifier import initialize_classifier
from src.utils.config import cfg

def load_class_names(path=Path("models/checkpoints/class_names.json")):
    if not Path(path).exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "class_names" in data:
        return data["class_names"]
    if isinstance(data, list):
        return data
    # fallback: dict of index->name
    try:
        items = sorted(data.items(), key=lambda x: int(x[0]))
        return [v for _, v in items]
    except Exception:
        return list(data.values())

def export_onnx(
    ckpt_path="models/checkpoints/classifier.pth",
    out_path="models/exports/classifier.onnx",
    opset=13,
    input_size=(1, 3, 224, 224),
    dynamic_batch=True,
):
    device = torch.device("cpu")  # export on cpu for portability

    # figure out num_classes from saved class_names.json (preferred) or config fallback
    class_names = load_class_names(Path("models/checkpoints/class_names.json"))
    if class_names:
        num_classes = len(class_names)
    else:
        num_classes = cfg.get("training", {}).get("num_classes", 2)

    model, _ = initialize_classifier(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    dummy_input = torch.randn(*input_size, dtype=torch.float32, device=device)

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    print(f"Exporting ONNX to: {out_path} (opset={opset})")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # Basic ONNX model validation
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model file saved and checker passed.")

    # Quick numerical sanity check using ONNXRuntime vs PyTorch
    ort_sess = ort.InferenceSession(str(out_path))
    def to_numpy(t):
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t)

    # PyTorch output
    with torch.no_grad():
        torch_out = model(dummy_input).cpu().numpy()

    # ONNXRuntime output
    ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_sess.run(None, ort_inputs)
    ort_out = np.array(ort_outs[0])

    # Compare shapes and compute max absolute diff
    print("PyTorch output shape:", torch_out.shape, "ONNX output shape:", ort_out.shape)
    max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
    mean_abs_diff = float(np.mean(np.abs(torch_out - ort_out)))
    print(f"Max abs diff: {max_abs_diff:.6g}, Mean abs diff: {mean_abs_diff:.6g}")

    # Tolerance sanity: typical diffs are small (1e-5 .. 1e-3). Warn if large.
    if max_abs_diff > 1e-2:
        print("WARNING: large difference between PyTorch and ONNX outputs. Investigate opset or export settings.")

    return str(out_path), {"max_abs_diff": max_abs_diff, "mean_abs_diff": mean_abs_diff}

if __name__ == "__main__":
    onnx_path, stats = export_onnx()
    print("Export complete:", onnx_path)
    print("Stats:", stats)

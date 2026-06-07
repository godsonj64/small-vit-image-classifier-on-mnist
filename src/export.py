"""Export a trained model to ONNX and TorchScript formats."""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from model import build_model
from utils import load_config, get_device, load_checkpoint


def export_onnx(
    model: nn.Module,
    save_path: str,
    image_size: int,
    in_channels: int,
    batch_size: int = 1,
) -> None:
    """Trace the model and save it in ONNX format."""
    model.eval()
    dummy = torch.zeros(batch_size, in_channels, image_size, image_size, device=next(model.parameters()).device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        save_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[INFO] ONNX model saved to {save_path}")


def export_torchscript(
    model: nn.Module,
    save_path: str,
    image_size: int,
    in_channels: int,
) -> None:
    """Trace the model and save it as a TorchScript file."""
    model.eval()
    dummy = torch.zeros(1, in_channels, image_size, image_size, device=next(model.parameters()).device)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
    traced.save(save_path)
    print(f"[INFO] TorchScript model saved to {save_path}")


def verify_onnx(save_path: str, image_size: int, in_channels: int) -> None:
    """Quick sanity check: run one forward pass via ONNXRuntime."""
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(save_path, providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, in_channels, image_size, image_size), dtype=np.float32)
        out = sess.run(["logits"], {"input": dummy})
        print(f"[INFO] ONNX verification passed. Output shape: {out[0].shape}")
    except Exception as exc:
        print(f"[WARN] ONNX verification skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["outputs"]["best_model"]
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()

    image_size: int = cfg["dataset"]["image_size"]
    in_channels: int = cfg["model"]["in_channels"]

    # ONNX
    export_onnx(
        model,
        cfg["outputs"]["onnx_model"],
        image_size,
        in_channels,
    )
    verify_onnx(cfg["outputs"]["onnx_model"], image_size, in_channels)

    # TorchScript
    export_torchscript(
        model,
        cfg["outputs"]["torchscript_model"],
        image_size,
        in_channels,
    )

    print("[INFO] Export complete.")


if __name__ == "__main__":
    main()

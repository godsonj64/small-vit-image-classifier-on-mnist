"""Export the trained model to ONNX and TorchScript formats."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import build_model
from src.utils import load_config, load_checkpoint


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, path: str, opset: int):
    """Trace and save the model in ONNX format."""
    torch.onnx.export(
        model,
        dummy_input,
        path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"ONNX model saved → {path}")

    # Quick sanity check
    try:
        import onnx
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")
    except ImportError:
        print("onnx package not found; skipping model check.")


def export_torchscript(model: torch.nn.Module, dummy_input: torch.Tensor, path: str):
    """Trace and save the model in TorchScript format."""
    scripted = torch.jit.trace(model, dummy_input)
    scripted.save(path)
    print(f"TorchScript model saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="Export trained ViT model")
    parser.add_argument("--config",     default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None,                   help="Path to .pth checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ckpt_path = args.checkpoint or os.path.join(
        cfg["paths"]["output_dir"], cfg["paths"]["checkpoint_name"]
    )
    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device     = torch.device("cpu")   # export on CPU for portability
    model      = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()

    img_size   = cfg["dataset"]["image_size"]
    in_ch      = cfg["model"]["in_channels"]
    batch_size = 1
    dummy      = torch.zeros(batch_size, in_ch, img_size, img_size, device=device)

    export_cfg  = cfg["export"]
    opset       = export_cfg.get("opset_version", 17)

    onnx_path = os.path.join(output_dir, export_cfg["onnx_filename"])
    ts_path   = os.path.join(output_dir, export_cfg["torchscript_filename"])

    export_onnx(model, dummy, onnx_path, opset)
    export_torchscript(model, dummy, ts_path)

    print("\nExport complete.")
    print(f"  ONNX        : {onnx_path}")
    print(f"  TorchScript : {ts_path}")


if __name__ == "__main__":
    main()

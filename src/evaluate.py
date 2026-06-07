"""Evaluation script — computes accuracy and macro F1 on the test split."""

import argparse
import os
import sys

import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import get_dataloaders
from src.model   import build_model
from src.utils   import load_config, load_checkpoint


@torch.no_grad()
def run_evaluation(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Small ViT on MNIST test set")
    parser.add_argument("--config",     default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None,                   help="Path to .pth checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ckpt_path = args.checkpoint or os.path.join(
        cfg["paths"]["output_dir"], cfg["paths"]["checkpoint_name"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoint: {ckpt_path}")

    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device)

    _, _, test_loader = get_dataloaders(cfg)

    print("Running evaluation on test set …")
    y_true, y_pred = run_evaluation(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")

    print("\n" + "=" * 50)
    print(f"Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Test F1 (macro): {f1:.4f}")
    print("=" * 50)

    class_names = [str(i) for i in range(cfg["model"]["num_classes"])]
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()

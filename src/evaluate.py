"""Evaluate a trained checkpoint on the test set.

Outputs:
  - Accuracy
  - Macro-averaged F1
  - Confusion matrix (saved as PNG)
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)

from dataset import build_dataloaders
from model import build_model
from utils import load_config, get_device, load_checkpoint


@torch.no_grad()
def run_inference(model, loader, device):
    """Collect all predictions and ground-truth labels from a DataLoader."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Plot and save a confusion matrix as a PNG file."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained MNIST model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    checkpoint_path = args.checkpoint or cfg["outputs"]["best_model"]
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    model = build_model(cfg).to(device)
    load_checkpoint(model, checkpoint_path, device)

    _, _, test_loader = build_dataloaders(cfg)

    print("[INFO] Running inference on test set...")
    labels, preds = run_inference(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)
    class_names = [str(i) for i in range(cfg["dataset"]["num_classes"])]

    print("\n" + "=" * 50)
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Macro F1 Score: {f1:.4f}")
    print("=" * 50)
    print("\nPer-class report:")
    print(classification_report(labels, preds, target_names=class_names))

    plot_confusion_matrix(cm, class_names, cfg["outputs"]["confusion_matrix"])


if __name__ == "__main__":
    main()

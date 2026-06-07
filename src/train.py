"""Training loop for ViT-Tiny / CNN on MNIST."""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from dataset import build_dataloaders
from model import build_model
from utils import load_config, seed_everything, get_device, AverageMeter, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
) -> float:
    """Run one full pass over the training set and return average loss."""
    model.train()
    loss_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate on the validation set and return (avg_loss, accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return loss_meter.avg, accuracy


def main():
    parser = argparse.ArgumentParser(description="Train ViT-Tiny on MNIST")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["training"]["seed"])
    device = get_device()

    os.makedirs(cfg["outputs"]["dir"], exist_ok=True)

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Building dataloaders for dataset: {cfg['dataset']['name']}")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    print(f"[INFO] Building model: {cfg['model']['name']}")
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["training"]["label_smoothing"])
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps = len(train_loader) * cfg["training"]["warmup_epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg["training"]["learning_rate"],
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
        anneal_strategy="cos",
    )

    epochs = cfg["training"]["epochs"]
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        _, val_acc = validate(model, val_loader, criterion, device)

        # ── Canonical log line (parsed by runner) ──────────────────────────
        print(f"epoch {epoch}/{epochs} loss={train_loss:.4f} val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, val_acc, cfg["outputs"]["best_model"])

    # Always save last checkpoint
    save_checkpoint(model, optimizer, epochs, val_acc, cfg["outputs"]["last_model"])
    print(f"[INFO] Training complete. Best val_acc={best_val_acc:.4f}")
    print(f"[INFO] Best checkpoint saved to {cfg['outputs']['best_model']}")


if __name__ == "__main__":
    main()

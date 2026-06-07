"""Training script for the Small ViT MNIST classifier."""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import get_dataloaders
from src.model   import build_model
from src.utils   import (
    set_seed,
    load_config,
    save_checkpoint,
    AverageMeter,
    compute_accuracy,
)


def build_optimizer(model: nn.Module, cfg: dict):
    train_cfg = cfg["training"]
    opt_name  = train_cfg["optimizer"].lower()
    lr        = train_cfg["learning_rate"]
    wd        = train_cfg["weight_decay"]

    if opt_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg  = cfg["training"]
    sched_name = train_cfg["scheduler"].lower()
    epochs     = train_cfg["epochs"]
    warmup     = train_cfg.get("warmup_epochs", 0)

    if sched_name == "cosine":
        # Linear warmup then cosine decay
        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(max(1, warmup))
            progress = float(epoch - warmup) / float(max(1, epochs - warmup))
            return 0.5 * (1.0 + torch.tensor(progress * 3.14159).cos().item())
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif sched_name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


def train_one_epoch(
    model, loader, criterion, optimizer, device, grad_clip: float
) -> float:
    model.train()
    loss_meter = AverageMeter()

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total   = 0

    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss_meter.update(loss.item(), images.size(0))

        preds    = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    acc = correct / total
    return loss_meter.avg, acc


def main():
    parser = argparse.ArgumentParser(description="Train Small ViT on MNIST")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = cfg["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['architecture']} | Trainable params: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg["training"].get("label_smoothing", 0.0)
    )
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    grad_clip = cfg["training"].get("grad_clip", 0.0)

    # ── Training loop ─────────────────────────────────────────────────────────
    epochs    = cfg["training"]["epochs"]
    best_acc  = 0.0
    ckpt_path = os.path.join(output_dir, cfg["paths"]["checkpoint_name"])

    log_path = cfg["paths"].get("log_file", os.path.join(output_dir, "train.log"))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "w") as log_f:
        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, grad_clip
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            if scheduler is not None:
                scheduler.step()

            # ── Required log line (parsed by the runner) ───────────────────
            log_line = (
                f"epoch {epoch}/{epochs} "
                f"loss={train_loss:.4f} "
                f"val_acc={val_acc:.4f}"
            )
            print(log_line)
            log_f.write(log_line + "\n")
            log_f.flush()

            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(model, optimizer, epoch, val_acc, ckpt_path)

    print(f"\nTraining complete. Best val_acc={best_acc:.4f}")
    print(f"Best checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()

"""Shared utility functions for the ViT-MNIST project."""

import os
import random
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Slightly slower but fully deterministic on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch:     int,
    val_acc:   float,
    path:      str,
) -> None:
    """Save model weights, optimiser state, epoch, and best metric."""
    state = {
        "epoch":     epoch,
        "val_acc":   val_acc,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(
    model:  nn.Module,
    path:   str,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict:
    """Load a checkpoint and restore model (and optionally optimizer) weights."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks a running average of a scalar (e.g. loss) over mini-batches."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return top-1 accuracy as a Python float."""
    preds   = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a simple logger that writes to stdout (and optionally a file)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger

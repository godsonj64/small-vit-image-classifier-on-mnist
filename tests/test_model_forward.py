"""Unit tests for SmallViT and BaselineCNN forward / backward passes."""

import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model import SmallViT, BaselineCNN, build_model
from src.utils import load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────

BATCH = 4
IN_CH = 1
IMG   = 28
CLASSES = 10


@pytest.fixture
def sample_input():
    return torch.randn(BATCH, IN_CH, IMG, IMG)


@pytest.fixture
def vit_model():
    return SmallViT(
        image_size=IMG, patch_size=7, in_channels=IN_CH,
        num_classes=CLASSES, dim=128, depth=4, heads=4,
        mlp_dim=256, dropout=0.0, emb_dropout=0.0,
    )


@pytest.fixture
def cnn_model():
    return BaselineCNN(in_channels=IN_CH, num_classes=CLASSES)


# ── SmallViT tests ────────────────────────────────────────────────────────────

def test_vit_output_shape(vit_model, sample_input):
    """SmallViT must output (batch, num_classes) logits."""
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(sample_input)
    assert out.shape == (BATCH, CLASSES), f"Expected {(BATCH, CLASSES)}, got {out.shape}"


def test_vit_backward(vit_model, sample_input):
    """Gradients must flow through SmallViT without NaN or error."""
    vit_model.train()
    out  = vit_model(sample_input)
    loss = out.sum()
    loss.backward()
    for name, p in vit_model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


def test_vit_num_patches():
    """Patch embedding must produce the correct number of patches."""
    model = SmallViT(image_size=28, patch_size=7)
    # 28/7 = 4 patches per dimension → 16 total
    assert model.patch_embed.num_patches == 16


def test_vit_cls_token_shape(vit_model, sample_input):
    """The CLS token should not affect the output tensor shape."""
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(sample_input)
    assert out.shape[1] == CLASSES


# ── BaselineCNN tests ─────────────────────────────────────────────────────────

def test_cnn_output_shape(cnn_model, sample_input):
    """BaselineCNN must output (batch, num_classes) logits."""
    cnn_model.eval()
    with torch.no_grad():
        out = cnn_model(sample_input)
    assert out.shape == (BATCH, CLASSES), f"Expected {(BATCH, CLASSES)}, got {out.shape}"


def test_cnn_backward(cnn_model, sample_input):
    """Gradients must flow through BaselineCNN without NaN or error."""
    cnn_model.train()
    out  = cnn_model(sample_input)
    loss = out.sum()
    loss.backward()
    for name, p in cnn_model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"


# ── Factory tests ─────────────────────────────────────────────────────────────

def test_build_model_vit():
    """build_model must return a SmallViT when architecture=small_vit."""
    cfg = load_config("configs/default.yaml")
    cfg["model"]["architecture"] = "small_vit"
    model = build_model(cfg)
    assert isinstance(model, SmallViT)


def test_build_model_cnn():
    """build_model must return a BaselineCNN when architecture=baseline_cnn."""
    cfg = load_config("configs/default.yaml")
    cfg["model"]["architecture"] = "baseline_cnn"
    model = build_model(cfg)
    assert isinstance(model, BaselineCNN)


def test_build_model_unknown():
    """build_model must raise ValueError for unknown architectures."""
    cfg = load_config("configs/default.yaml")
    cfg["model"]["architecture"] = "unknown_arch"
    with pytest.raises(ValueError):
        build_model(cfg)


# ── Consistency test ──────────────────────────────────────────────────────────

def test_vit_deterministic(vit_model, sample_input):
    """SmallViT in eval mode must return identical results on two forward passes."""
    vit_model.eval()
    with torch.no_grad():
        out1 = vit_model(sample_input)
        out2 = vit_model(sample_input)
    assert torch.allclose(out1, out2), "Non-deterministic forward pass detected"

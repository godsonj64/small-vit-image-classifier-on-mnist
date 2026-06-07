"""Unit tests for model forward passes and dataset utilities."""

from __future__ import annotations

import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch

from model import ViTTiny, SmallCNN, build_model
from utils import load_config, AverageMeter, count_parameters


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    """A small batch of fake MNIST images (B=4, C=1, H=28, W=28)."""
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def default_cfg():
    """Load the default config."""
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    return load_config(cfg_path)


# ──────────────────────────────────────────────────────────────────────────────
# ViT-Tiny tests
# ──────────────────────────────────────────────────────────────────────────────

class TestViTTiny:
    def test_output_shape(self, dummy_batch):
        model = ViTTiny(
            image_size=28, patch_size=4, in_channels=1,
            num_classes=10, dim=192, depth=6, num_heads=3, mlp_dim=384, dropout=0.0
        )
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_num_patches(self, dummy_batch):
        model = ViTTiny(image_size=28, patch_size=4, in_channels=1, num_classes=10)
        assert model.patch_embed.num_patches == 49  # (28/4)^2

    def test_gradient_flows(self, dummy_batch):
        model = ViTTiny(image_size=28, patch_size=4, in_channels=1, num_classes=10)
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_param_count(self):
        model = ViTTiny(
            image_size=28, patch_size=4, in_channels=1, num_classes=10,
            dim=192, depth=6, num_heads=3, mlp_dim=384
        )
        n = count_parameters(model)
        # Should be roughly 5 M for these settings
        assert 1_000_000 < n < 20_000_000, f"Unexpected param count: {n}"


# ──────────────────────────────────────────────────────────────────────────────
# SmallCNN tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSmallCNN:
    def test_output_shape(self, dummy_batch):
        model = SmallCNN(in_channels=1, num_classes=10)
        model.eval()
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_gradient_flows(self, dummy_batch):
        model = SmallCNN(in_channels=1, num_classes=10)
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_param_count(self):
        model = SmallCNN(in_channels=1, num_classes=10)
        n = count_parameters(model)
        assert n < 500_000, f"CNN baseline too large: {n} params"


# ──────────────────────────────────────────────────────────────────────────────
# build_model factory tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBuildModel:
    def test_build_vit(self, default_cfg):
        default_cfg["model"]["name"] = "vit_tiny"
        model = build_model(default_cfg)
        assert isinstance(model, ViTTiny)

    def test_build_cnn(self, default_cfg):
        default_cfg["model"]["name"] = "cnn_baseline"
        model = build_model(default_cfg)
        assert isinstance(model, SmallCNN)

    def test_invalid_model(self, default_cfg):
        default_cfg["model"]["name"] = "nonexistent_model"
        with pytest.raises(ValueError):
            build_model(default_cfg)


# ──────────────────────────────────────────────────────────────────────────────
# AverageMeter tests
# ──────────────────────────────────────────────────────────────────────────────

class TestAverageMeter:
    def test_basic(self):
        meter = AverageMeter()
        meter.update(1.0, 1)
        meter.update(3.0, 1)
        assert abs(meter.avg - 2.0) < 1e-6

    def test_weighted(self):
        meter = AverageMeter()
        meter.update(2.0, 4)   # contributes 8
        meter.update(6.0, 4)   # contributes 24  → total 32 / 8 = 4.0
        assert abs(meter.avg - 4.0) < 1e-6

    def test_reset(self):
        meter = AverageMeter()
        meter.update(5.0, 1)
        meter.reset()
        assert meter.avg == 0.0 and meter.count == 0

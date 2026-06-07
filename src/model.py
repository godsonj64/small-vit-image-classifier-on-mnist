"""Model definitions: ViT-Tiny (main) and Small CNN (baseline)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Vision Transformer – Tiny
# ──────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split an image into fixed-size patches and linearly project them."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size {image_size} must be divisible by patch_size {patch_size}"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, H, W) -> (B, N, D)
        x = self.proj(x)          # (B, D, H/P, W/P)
        x = x.flatten(2)          # (B, D, N)
        x = x.transpose(1, 2)     # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional dropout."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """A single Transformer encoder block (pre-norm style)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTiny(nn.Module):
    """Vision Transformer – Tiny, built from scratch for MNIST.

    Default hyper-params (from config):
        image_size  = 28
        patch_size  = 4   → 49 patches
        dim         = 192
        depth       = 6
        num_heads   = 3
        mlp_dim     = 384
        dropout     = 0.1
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                       # (B, N+1, D)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]                                          # CLS token
        return self.head(x)


# ──────────────────────────────────────────────────────────────────────────────
# Small CNN Baseline
# ──────────────────────────────────────────────────────────────────────────────

class SmallCNN(nn.Module):
    """Lightweight 3-block CNN baseline trained from scratch on MNIST.

    Architecture:
        Conv(1→32) → BN → ReLU → MaxPool
        Conv(32→64) → BN → ReLU → MaxPool
        Conv(64→128) → BN → ReLU → AdaptiveAvgPool
        FC(128 → 10)
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, **_):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14×14
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 7×7
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),     # 1×1
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the correct model from the config dict."""
    model_cfg = cfg["model"]
    name: str = model_cfg["name"]
    kwargs = {
        "image_size": cfg["dataset"]["image_size"],
        "in_channels": model_cfg["in_channels"],
        "num_classes": model_cfg["num_classes"],
    }
    if name == "vit_tiny":
        kwargs.update(
            patch_size=model_cfg["patch_size"],
            dim=model_cfg["dim"],
            depth=model_cfg["depth"],
            num_heads=model_cfg["num_heads"],
            mlp_dim=model_cfg["mlp_dim"],
            dropout=model_cfg["dropout"],
        )
        return ViTTiny(**kwargs)
    elif name == "cnn_baseline":
        return SmallCNN(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {name!r}")

"""Model definitions: SmallViT (primary) and BaselineCNN (comparison)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Helper modules
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to embedding dim."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, dim: int):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)  ->  (B, num_patches, dim)
        x = self.projection(x)          # (B, dim, H/P, W/P)
        x = x.flatten(2)               # (B, dim, num_patches)
        x = x.transpose(1, 2)          # (B, num_patches, dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention block."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads   = heads
        self.scale   = (dim // heads) ** -0.5
        self.qkv     = nn.Linear(dim, dim * 3, bias=False)
        self.proj    = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        H       = self.heads
        head_dim = C // H

        qkv = self.qkv(x).reshape(B, N, 3, H, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)         # each (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """One Transformer encoder block: LayerNorm → MHSA → residual → LayerNorm → MLP → residual."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Primary model: Small Vision Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SmallViT(nn.Module):
    """
    Lightweight Vision Transformer for MNIST.

    With default config (patch_size=7, dim=128, depth=4, heads=4):
      - 16 patches per image
      - ~1.2 M parameters
    """

    def __init__(
        self,
        image_size:  int = 28,
        patch_size:  int = 7,
        in_channels: int = 1,
        num_classes: int = 10,
        dim:         int = 128,
        depth:       int = 4,
        heads:       int = 4,
        mlp_dim:     int = 256,
        dropout:     float = 0.1,
        emb_dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        num_patches      = self.patch_embed.num_patches

        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.transformer = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm       = nn.LayerNorm(dim)
        self.head       = nn.Linear(dim, num_classes)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, dim)

        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, dim)
        x   = torch.cat([cls, x], dim=1)                 # (B, N+1, dim)
        x   = x + self.pos_embed
        x   = self.emb_dropout(x)

        x   = self.transformer(x)                        # (B, N+1, dim)
        x   = self.norm(x)
        cls_out = x[:, 0]                                # (B, dim)
        return self.head(cls_out)                        # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline model: Small CNN
# ─────────────────────────────────────────────────────────────────────────────

class BaselineCNN(nn.Module):
    """
    Simple 3-block convolutional network trained from scratch.
    Serves as the baseline to compare against SmallViT.
    ~93 K parameters.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28 -> 14
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14 -> 7
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 7 -> 1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """Instantiate the model specified in cfg['model']['architecture']."""
    model_cfg = cfg["model"]
    arch      = model_cfg["architecture"]

    if arch == "small_vit":
        return SmallViT(
            image_size  = model_cfg["image_size"],
            patch_size  = model_cfg["patch_size"],
            in_channels = model_cfg["in_channels"],
            num_classes = model_cfg["num_classes"],
            dim         = model_cfg["dim"],
            depth       = model_cfg["depth"],
            heads       = model_cfg["heads"],
            mlp_dim     = model_cfg["mlp_dim"],
            dropout     = model_cfg["dropout"],
            emb_dropout = model_cfg["emb_dropout"],
        )
    elif arch == "baseline_cnn":
        return BaselineCNN(
            in_channels = model_cfg["in_channels"],
            num_classes = model_cfg["num_classes"],
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

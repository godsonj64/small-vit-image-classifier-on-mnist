"""Dataset loading and preprocessing for MNIST (and generic ImageFolder)."""

from __future__ import annotations

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


def get_mnist_transforms(train: bool, image_size: int = 28) -> transforms.Compose:
    """Return appropriate torchvision transforms for MNIST."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_image_folder_transforms(train: bool, image_size: int = 28) -> transforms.Compose:
    """Generic ImageFolder transforms (single-channel assumed)."""
    if train:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def build_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from config.

    Args:
        cfg: Full config dict (loaded from YAML).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    name: str = dataset_cfg["name"]
    data_dir: str = dataset_cfg["data_dir"]
    image_size: int = dataset_cfg["image_size"]
    num_workers: int = dataset_cfg["num_workers"]
    pin_memory: bool = dataset_cfg["pin_memory"]
    batch_size: int = training_cfg["batch_size"]
    val_split: float = training_cfg["val_split"]
    seed: int = training_cfg["seed"]

    os.makedirs(data_dir, exist_ok=True)

    if name == "mnist":
        train_full = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=get_mnist_transforms(train=True, image_size=image_size),
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=get_mnist_transforms(train=False, image_size=image_size),
        )
    elif name == "image_folder":
        train_full = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=get_image_folder_transforms(train=True, image_size=image_size),
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "test"),
            transform=get_image_folder_transforms(train=False, image_size=image_size),
        )
    else:
        raise ValueError(f"Unknown dataset name: {name!r}")

    # Train / val split
    n_total = len(train_full)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        train_full, [n_train, n_val], generator=generator
    )

    # For validation we want the non-augmented transform. We achieve this by
    # wrapping the Subset in a small adapter that swaps the transform.
    if name == "mnist":
        val_transform = get_mnist_transforms(train=False, image_size=image_size)
    else:
        val_transform = get_image_folder_transforms(train=False, image_size=image_size)

    val_dataset = _TransformSubset(train_full, val_dataset.indices, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


class _TransformSubset(torch.utils.data.Dataset):
    """A Subset wrapper that applies a *different* transform than the parent."""

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_transform = self.base_dataset.transform
        self.base_dataset.transform = self.transform
        sample = self.base_dataset[self.indices[idx]]
        self.base_dataset.transform = original_transform
        return sample

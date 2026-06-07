"""Dataset loading and preprocessing for MNIST (and generic image_folder)."""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_transforms(image_size: int, split: str) -> transforms.Compose:
    """Return the appropriate transform pipeline for train / val / test."""
    mean = (0.1307,)   # MNIST channel-wise mean
    std  = (0.3081,)   # MNIST channel-wise std

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader) based on the config dict.

    Supports:
      - cfg['dataset']['name'] == 'mnist'        -> torchvision MNIST
      - cfg['dataset']['name'] == 'image_folder' -> ImageFolder layout
    """
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    name       = dataset_cfg["name"]
    data_dir   = dataset_cfg["data_dir"]
    image_size = dataset_cfg["image_size"]
    batch_size = training_cfg["batch_size"]
    num_workers = dataset_cfg.get("num_workers", 4)
    pin_memory  = dataset_cfg.get("pin_memory", True)

    if name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=get_transforms(image_size, "train"),
        )
        # Split off 10 % of training data as validation
        val_size   = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        # Apply val transforms to the validation split
        val_dataset.dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=get_transforms(image_size, "val"),
        )

        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=get_transforms(image_size, "test"),
        )

    elif name == "image_folder":
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "train"),
            transform=get_transforms(image_size, "train"),
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "val"),
            transform=get_transforms(image_size, "val"),
        )
        test_dataset = val_dataset  # re-use val as test for image_folder
    else:
        raise ValueError(f"Unknown dataset name: {name}")

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

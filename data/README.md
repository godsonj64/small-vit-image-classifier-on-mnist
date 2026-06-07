# Dataset: MNIST

## Overview
MNIST is a classic benchmark dataset of 70,000 grayscale 28×28 images of handwritten digits (0–9).
- **Training set:** 60,000 images
- **Test set:** 10,000 images
- **Classes:** 10 (digits 0 through 9)

## Automatic Download
The dataset is downloaded automatically by `torchvision` the first time you run `src/train.py`. It will be saved under `./data/MNIST/`.

```
data/
└── MNIST/
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

## Manual Download
If you are in an air-gapped environment, download the four binary files from:
```
http://yann.lecun.com/exdb/mnist/
```
and place them in `data/MNIST/raw/`.

## Image Folder Format (custom data)
If you want to use your own digit images instead of MNIST, organise them as:
```
data/
├── train/
│   ├── 0/   ← PNG/JPG images of digit 0
│   ├── 1/
│   └── ...
└── val/
    ├── 0/
    ├── 1/
    └── ...
```
Then set `dataset.name: image_folder` in `configs/default.yaml`.

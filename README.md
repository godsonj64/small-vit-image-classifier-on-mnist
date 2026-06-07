# Small ViT Image Classifier on MNIST

This project trains a lightweight **Vision Transformer (ViT-Tiny)** from scratch on the classic **MNIST** handwritten digit dataset, classifying grayscale 28×28 images into one of 10 digit categories (0–9). A small **CNN baseline** is also included for comparison.

## Project Structure

```
.
├── configs/
│   └── default.yaml        # All hyperparameters and paths
├── data/
│   └── README.md           # How dataset is organised on disk
├── scripts/
│   └── run_train.sh        # One-click training script
├── src/
│   ├── dataset.py          # Dataset loading & augmentation
│   ├── model.py            # ViT-Tiny & CNN baseline definitions
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Metrics: accuracy, F1, confusion matrix
│   ├── export.py           # ONNX & TorchScript export
│   └── utils.py            # Shared helpers (seeding, logging, etc.)
├── tests/
│   └── test_model_forward.py
├── Dockerfile
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train
```bash
bash scripts/run_train.sh
# or directly:
python src/train.py --config configs/default.yaml
```

### 3. Evaluate
```bash
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

### 4. Export
```bash
python src/export.py --config configs/default.yaml --checkpoint outputs/best_model.pt
```

Exported artefacts land in `outputs/`:
- `model.onnx`
- `model_torchscript.pt`

## Docker
```bash
docker build -t vit-mnist .
docker run --rm -v $(pwd)/outputs:/app/outputs vit-mnist
```

## Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| ViT-Tiny (main) | ~5 M | Patch size 4, depth 6 |
| Small CNN (baseline) | ~200 K | 3 conv blocks + FC |

## Metrics Reported
- **Accuracy** – overall correct classifications
- **F1 Score** – macro-averaged F1 across 10 classes
- **Confusion Matrix** – saved as `outputs/confusion_matrix.png`

## Dataset
MNIST is downloaded automatically via `torchvision` on first run into `data/mnist/`.

# Small ViT Image Classifier on MNIST

This project trains a lightweight Vision Transformer (ViT) to classify handwritten digits (0–9) from the MNIST dataset. A small CNN baseline is also included for comparison.

## Project Structure

```
.
├── configs/
│   └── default.yaml        # All hyperparameters and paths
├── data/
│   └── README.md           # How to obtain / organise the dataset
├── scripts/
│   └── run_train.sh        # One-click training script
├── src/
│   ├── dataset.py          # Data loading and augmentation
│   ├── model.py            # ViT and CNN baseline architectures
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation (accuracy + F1)
│   ├── export.py           # ONNX and TorchScript export
│   └── utils.py            # Shared helpers
├── tests/
│   └── test_model_forward.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare data
See `data/README.md`. The dataset will be auto-downloaded by torchvision the first time you run training.

### 3. Train
```bash
bash scripts/run_train.sh
```
Or manually:
```bash
python src/train.py --config configs/default.yaml
```

### 4. Evaluate
```bash
python src/evaluate.py --config configs/default.yaml --checkpoint outputs/best_model.pth
```

### 5. Export
```bash
python src/export.py --config configs/default.yaml --checkpoint outputs/best_model.pth
```
This produces `outputs/model.onnx` and `outputs/model_torchscript.pt`.

## Docker
```bash
docker build -t vit-mnist .
docker run --rm -v $(pwd)/outputs:/app/outputs vit-mnist
```

## Metrics
- **Accuracy** – fraction of correctly classified digits
- **F1 Score** – macro-averaged F1 across all 10 classes

## Models
| Model | Description |
|-------|-------------|
| SmallViT | Lightweight Vision Transformer (4 layers, 4 heads, patch 7) |
| BaselineCNN | Simple 3-layer CNN trained from scratch |

## Expected Results
| Model | Val Accuracy | Val F1 |
|-------|-------------|--------|
| BaselineCNN | ~98.5% | ~0.985 |
| SmallViT | ~99.2% | ~0.992 |

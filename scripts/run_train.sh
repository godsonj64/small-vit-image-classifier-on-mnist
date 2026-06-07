#!/usr/bin/env bash
# run_train.sh — Train the Small ViT on MNIST, then evaluate and export.

set -euo pipefail

CONFIG="configs/default.yaml"
OUTPUT_DIR="outputs"
CHECKPOINT="${OUTPUT_DIR}/best_model.pth"

echo "=========================================="
echo " Small ViT MNIST — Training Pipeline"
echo "=========================================="
echo "Config     : ${CONFIG}"
echo "Output dir : ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── Step 1: Train ────────────────────────────────────────────────────────────
echo "[1/3] Starting training …"
python src/train.py --config "${CONFIG}"
echo ""

# ── Step 2: Evaluate ─────────────────────────────────────────────────────────
echo "[2/3] Evaluating best checkpoint …"
python src/evaluate.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
echo ""

# ── Step 3: Export ───────────────────────────────────────────────────────────
echo "[3/3] Exporting model …"
python src/export.py --config "${CONFIG}" --checkpoint "${CHECKPOINT}"
echo ""

echo "=========================================="
echo " Pipeline complete!"
echo "  Checkpoint  : ${CHECKPOINT}"
echo "  ONNX        : ${OUTPUT_DIR}/model.onnx"
echo "  TorchScript : ${OUTPUT_DIR}/model_torchscript.pt"
echo "=========================================="

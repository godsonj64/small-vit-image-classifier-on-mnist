#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# run_train.sh  –  Train → Evaluate → Export pipeline for ViT-MNIST
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

CONFIG="$PROJECT_ROOT/configs/default.yaml"
CHECKPOINT="$PROJECT_ROOT/outputs/best_model.pt"

echo "=========================================="
echo "  ViT-Tiny MNIST — Training Pipeline"
echo "=========================================="

# ── 1. Train ──────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 1/3] Training..."
python "$PROJECT_ROOT/src/train.py" --config "$CONFIG"

# ── 2. Evaluate ───────────────────────────────────────────────────────────────
echo ""
echo "[STEP 2/3] Evaluating on test set..."
python "$PROJECT_ROOT/src/evaluate.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"

# ── 3. Export ─────────────────────────────────────────────────────────────────
echo ""
echo "[STEP 3/3] Exporting model (ONNX + TorchScript)..."
python "$PROJECT_ROOT/src/export.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"

echo ""
echo "=========================================="
echo "  Pipeline complete! Artefacts in outputs/"
echo "=========================================="

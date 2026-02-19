#!/usr/bin/env bash
# publish_hf.sh — Export checkpoint and push to HuggingFace Hub + Space
#
# Usage:
#   ./scripts/publish_hf.sh [model-size] [checkpoint]
#
#   model-size  xs/s/m/l/deep/xl  (default: s)
#   checkpoint  path to .bin file  (default: checkpoint_best.bin)
#
# Examples:
#   ./scripts/publish_hf.sh
#   ./scripts/publish_hf.sh s checkpoint_best.bin
#   ./scripts/publish_hf.sh m checkpoint.bin

set -euo pipefail

MODEL_SIZE="${1:-s}"
CHECKPOINT="${2:-checkpoint_best.bin}"
MODEL_REPO="MonumentalSystems/randygpt-${MODEL_SIZE}"
SPACE_REPO="MonumentalSystems/randygpt-space"
EXPORT_DIR="hf_export"
VOCAB="vocab.json"

echo "==> randyGPT publish"
echo "    checkpoint : ${CHECKPOINT}"
echo "    model size : ${MODEL_SIZE}"
echo "    model repo : ${MODEL_REPO}"
echo "    space repo : ${SPACE_REPO}"
echo ""

# 1. Export checkpoint to safetensors + config + tokenizer
echo "[1/3] Exporting checkpoint..."
python3 scripts/export_hf.py \
    --checkpoint "${CHECKPOINT}" \
    --vocab      "${VOCAB}" \
    --output     "${EXPORT_DIR}" \
    --model-size "${MODEL_SIZE}" \
    --repo       "${MODEL_REPO}"

echo ""

# 2. Upload model weights to Hub
echo "[2/3] Uploading model to ${MODEL_REPO}..."
hf upload "${MODEL_REPO}" "${EXPORT_DIR}" . --repo-type model

echo ""

# 3. Upload Space files
echo "[3/4] Uploading Space to ${SPACE_REPO}..."
hf upload "${SPACE_REPO}" spaces . --repo-type space

echo ""

# 4. Restart Space so it reloads fresh model weights from Hub
echo "[4/4] Restarting Space..."
python3 -c "
from huggingface_hub import HfApi
HfApi().restart_space('${SPACE_REPO}')
print('Space restart requested.')
"

echo ""
echo "==> Done!"
echo "    Model : https://huggingface.co/${MODEL_REPO}"
echo "    Space : https://huggingface.co/spaces/${SPACE_REPO}"

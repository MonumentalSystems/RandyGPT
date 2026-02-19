#!/usr/bin/env bash
# publish_hf.sh — Export checkpoint and push to HuggingFace Hub + Space
#
# Usage:
#   ./scripts/publish_hf.sh [model-size] [checkpoint] [--restart]
#
#   model-size  xs/s/m/l/ds/deep/xl  (default: s)
#   checkpoint  path to .bin file    (default: checkpoint_best.bin)
#   --restart   force full container restart instead of hot-reload
#
# Examples:
#   ./scripts/publish_hf.sh                              # weights only, hot-reload
#   ./scripts/publish_hf.sh s checkpoint_best.bin        # weights only, hot-reload
#   ./scripts/publish_hf.sh s checkpoint_best.bin --restart  # force restart

set -euo pipefail

MODEL_SIZE="${1:-s}"
CHECKPOINT="${2:-checkpoint_best.bin}"
FORCE_RESTART=false
for arg in "$@"; do [[ "$arg" == "--restart" ]] && FORCE_RESTART=true; done
MODEL_REPO="MonumentalSystems/randygpt-${MODEL_SIZE}"
SPACE_REPO="MonumentalSystems/randygpt-${MODEL_SIZE}-space"
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

# 3. Upload Space files (create space if it doesn't exist)
echo "[3/4] Uploading Space to ${SPACE_REPO}..."
hf repo create "${SPACE_REPO}" --repo-type space --space-sdk docker --exist-ok
# Inject MODEL_REPO env var into Space metadata
python3 -c "
from huggingface_hub import HfApi
HfApi().add_space_variable('${SPACE_REPO}', 'MODEL_REPO', '${MODEL_REPO}')
print('Set MODEL_REPO=${MODEL_REPO}')
"
SPACE_UPLOAD_OUT=$(hf upload "${SPACE_REPO}" spaces . --repo-type space 2>&1)
echo "${SPACE_UPLOAD_OUT}"
SPACE_FILES_CHANGED=true
if echo "${SPACE_UPLOAD_OUT}" | grep -q "No files have been modified"; then
    SPACE_FILES_CHANGED=false
fi

echo ""

# 4. Restart or hot-reload depending on whether Space code changed
SPACE_URL="https://$(echo ${SPACE_REPO} | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space"
if [ "${FORCE_RESTART}" = true ] || [ "${SPACE_FILES_CHANGED}" = true ]; then
    echo "[4/4] Space code changed — restarting container..."
    python3 -c "
from huggingface_hub import HfApi
HfApi().restart_space('${SPACE_REPO}')
print('Space restart requested.')
"
else
    echo "[4/4] Space code unchanged — hot-reloading weights..."
    if curl -sf -X POST "${SPACE_URL}/reload" -o /dev/null 2>/dev/null; then
        echo "Hot-reload successful."
    else
        echo "Space not responding, falling back to restart..."
        python3 -c "
from huggingface_hub import HfApi
HfApi().restart_space('${SPACE_REPO}')
print('Space restart requested.')
"
    fi
fi

echo ""
echo "==> Done!"
echo "    Model : https://huggingface.co/${MODEL_REPO}"
echo "    Space : https://huggingface.co/spaces/${SPACE_REPO}"

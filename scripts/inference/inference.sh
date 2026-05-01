#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/yourlogs}"
GPU_ID="${2:-${CUDA_VISIBLE_DEVICES:-0}}"
SAVE_VIS_INTERVAL="${SAVE_VIS_INTERVAL:-40}"

if [[ ! -f "${MODEL_DIR}/config.yaml" ]]; then
  echo "[ERR] Missing config.yaml under ${MODEL_DIR}" 1>&2
  exit 2
fi

if rg -q "v2xreal/test|dataset: ?v2xreal|dataset: 'v2xreal'" "${MODEL_DIR}/config.yaml"; then
  SAVE_VIS_INTERVAL="${SAVE_VIS_INTERVAL}" \
    bash ./scripts/inference_mc/run_v2xreal_official_mc.sh "${MODEL_DIR}" "${GPU_ID}"
else
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python ./opencood/tools/inference.py \
    --model_dir "${MODEL_DIR}" \
    --save_vis_interval "${SAVE_VIS_INTERVAL}"
fi

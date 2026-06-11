#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
OUTPUT="${RUN_ROOT}/diagnostics/latent_contract.json"
NUM_VIDEOS=32
DEVICE_ID=0
SEQ_LEN=8
TARGET_SIZE=518
MAX_FRAME_SPAN=32
LATENT_DIM=512
LATENT_GRID=18
NUM_WORKERS=2
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${GEOMETRY_AE_CKPT}" "geometry autoencoder checkpoint"

CUDA_VISIBLE_DEVICES="${DEVICE_ID}" python3 \
  "${PROJECT}/diagnose_latent_contract.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${GEOMETRY_AE_CKPT}" \
  --output "${OUTPUT}" \
  --num_videos "${NUM_VIDEOS}" \
  --seq_len "${SEQ_LEN}" \
  --target_size "${TARGET_SIZE}" \
  --max_frame_span "${MAX_FRAME_SPAN}" \
  --latent_dim "${LATENT_DIM}" \
  --latent_grid "${LATENT_GRID}" \
  --levels 4 11 17 23 \
  --num_workers "${NUM_WORKERS}"

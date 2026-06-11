#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
CHECKPOINT="${GEOMETRY_AE_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/evaluation/geometry_autoencoder"
NUM_VIDEOS=20
DEVICE_ID=0
SEQ_LEN=8
TARGET_SIZE=518
LATENT_DIM=512
LATENT_GRID=18
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${CHECKPOINT}" "geometry autoencoder checkpoint"

ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}" python3 \
  "${PROJECT}/inference_autoencoder.py" \
  --checkpoint "${CHECKPOINT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --seq_len "${SEQ_LEN}" \
  --target_size "${TARGET_SIZE}" \
  --latent_dim "${LATENT_DIM}" \
  --latent_grid "${LATENT_GRID}" \
  --levels 4 11 17 23 \
  --compute_psnr

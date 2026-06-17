#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/local_cuda.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
I0_CKPT="${I0_DECODER_CKPT}"
OUTPUT_DIR="${RUN_ROOT}/evaluation/temporal_reconstruction"
NUM_VIDEOS=4
NUM_WORKERS=0
SEQ_LEN=8
TARGET_SIZE=518
CLIP_DURATION_SECONDS=1.0
MAX_FRAME_SPAN=32
USE_EMA=1
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "I0 decoder checkpoint"

EXTRA_ARGS=()
if [[ "${USE_EMA}" -eq 1 ]]; then
  EXTRA_ARGS+=(--use_ema)
fi

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_IDS%%,*}" \
"${PYTHON_BIN}" "${PROJECT}/diagnose_temporal_reconstruction.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --num_workers "${NUM_WORKERS}" \
  --seq_len "${SEQ_LEN}" \
  --target_size "${TARGET_SIZE}" \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --max_frame_span "${MAX_FRAME_SPAN}" \
  --dtype fp16 \
  "${EXTRA_ARGS[@]}"

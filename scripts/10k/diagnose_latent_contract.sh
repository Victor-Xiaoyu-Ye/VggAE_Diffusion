#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
OUTPUT="${RUN_ROOT}/diagnostics/latent_contract.json"
NUM_VIDEOS=32
SEQ_LEN=8
TARGET_SIZE=518
MAX_FRAME_SPAN=32
CLIP_DURATION_SECONDS=1.0
LATENT_DIM=512
LATENT_GRID=18
NUM_WORKERS=2
DISABLE_TEMPORAL_MIXER=1
# -----------------------------------------------------------------------------

ensure_spatialvid_splits
require_file "${GEOMETRY_AE_CKPT}" "geometry autoencoder checkpoint"

EXTRA_ARGS=()
if [[ "${DISABLE_TEMPORAL_MIXER}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_temporal_mixer)
fi

"${PYTHON_BIN}" \
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
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --latent_dim "${LATENT_DIM}" \
  --latent_grid "${LATENT_GRID}" \
  --levels 4 11 17 23 \
  --num_workers "${NUM_WORKERS}" \
  "${EXTRA_ARGS[@]}"

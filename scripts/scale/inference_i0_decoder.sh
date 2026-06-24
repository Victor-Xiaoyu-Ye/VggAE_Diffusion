#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Manual I0 decoder inference on held-out SpatialVID clips.
# Output grids contain columns:
#   target / ae_full_z / ae_repeat_z0 / i0_full_z / i0_repeat_z0
# The repeat_z0 columns reveal whether the decoder is just copying I0.

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
OUTPUT_DIR="${SCALE_ROOT}/inference/i0_decoder"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/inference/i0_decoder"
NUM_VIDEOS=12
NUM_WORKERS=0
CLIP_DURATION_SECONDS=1.0
MAX_FRAME_SPAN=32
DTYPE=fp16
# -----------------------------------------------------------------------------

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

ensure_spatialvid_splits
ensure_local_checkpoint \
  "${AUTOENCODER_CKPT}" "${SCALE_GEOMETRY_AE_CKPT_URL}" \
  "scale geometry autoencoder checkpoint" \
  "${SCALE_GEOMETRY_AE_MIRROR_CKPT_URL}"
ensure_local_checkpoint \
  "${I0_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
  "scale I0 decoder checkpoint" \
  "${SCALE_I0_DECODER_MIRROR_CKPT_URL}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT}/diagnose_temporal_reconstruction.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --num_workers "${NUM_WORKERS}" \
  --seq_len 8 \
  --target_size 518 \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --max_frame_span "${MAX_FRAME_SPAN}" \
  --dtype "${DTYPE}"

"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
  "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}" --directory

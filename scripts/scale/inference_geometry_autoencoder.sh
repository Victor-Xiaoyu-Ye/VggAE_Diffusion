#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# ----------------------------- editable settings -----------------------------
# For a separate inference job, set CHECKPOINT_URL to the training job's OBS
# checkpoint path. If CHECKPOINT already exists locally, it is used directly.
CHECKPOINT="${SCALE_ROOT}/geometry_autoencoder/checkpoint_latest.pt"
CHECKPOINT_URL="${SCALE_REMOTE_ROOT}/geometry_autoencoder/checkpoint_latest.pt"
CHECKPOINT_MIRROR_URL="${SCALE_GEOMETRY_AE_MIRROR_CKPT_URL}"
OUTPUT_DIR="${SCALE_ROOT}/inference/geometry_autoencoder"
REMOTE_OUTPUT_DIR="${SCALE_REMOTE_ROOT}/inference/geometry_autoencoder"
NUM_VIDEOS=20
NUM_WORKERS=0
CLIP_DURATION_SECONDS=1.0
# -----------------------------------------------------------------------------

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

ensure_spatialvid_splits
ensure_local_checkpoint \
  "${CHECKPOINT}" "${CHECKPOINT_URL}" \
  "geometry autoencoder inference checkpoint" \
  "${CHECKPOINT_MIRROR_URL}"

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${PROJECT}/inference_autoencoder.py" \
  --checkpoint "${CHECKPOINT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --num_workers "${NUM_WORKERS}" \
  --clip_duration_seconds "${CLIP_DURATION_SECONDS}" \
  --compute_psnr

"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
  "${OUTPUT_DIR}" "${REMOTE_OUTPUT_DIR}" --directory

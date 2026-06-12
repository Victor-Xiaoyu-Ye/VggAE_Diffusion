#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
GENERATOR_CKPT="${SCALE_DIFFUSION_CKPT}"
OUT_DIR="${SCALE_ROOT}/samples"
REMOTE_OUT_DIR="${SCALE_REMOTE_ROOT}/samples"
SEED=42
NUM_STEPS=50
SOLVER="midpoint"
FPS=8

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi
validate_model_config
ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "scale I0 decoder checkpoint"
require_file "${GENERATOR_CKPT}" "scale diffusion checkpoint"

"${PYTHON_BIN}" "${PROJECT}/sample_compact_i0.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --diffusion_ckpt "${GENERATOR_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --solver "${SOLVER}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --dtype fp16

"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
  "${OUT_DIR}" "${REMOTE_OUT_DIR}" --directory

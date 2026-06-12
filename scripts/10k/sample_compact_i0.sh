#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
AUTOENCODER_CKPT="${GEOMETRY_AE_CKPT}"
I0_CKPT="${I0_DECODER_CKPT}"
GENERATOR_CKPT="${DIFFUSION_CKPT}"
OUT_DIR="${RUN_ROOT}/samples/compact_i0"
SEED=42
NUM_STEPS=50
SOLVER="midpoint"
FPS=8
# -----------------------------------------------------------------------------

validate_model_config
ensure_spatialvid_splits
require_file "${AUTOENCODER_CKPT}" "geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "I0 decoder checkpoint"
require_file "${GENERATOR_CKPT}" "diffusion checkpoint"

"${PYTHON_BIN}" \
  "${PROJECT}/sample_compact_i0.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --diffusion_ckpt "${GENERATOR_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --solver "${SOLVER}" \
  --seed "${SEED}" \
  --fps "${FPS}" \
  --dtype fp16

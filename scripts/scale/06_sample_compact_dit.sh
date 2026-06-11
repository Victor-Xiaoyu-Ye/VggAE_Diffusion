#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"

# ----------------------------- editable settings -----------------------------
REFERENCE_PATH="${I0_PATH}"
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
GENERATOR_CKPT="${SCALE_DIFFUSION_CKPT}"
OUT_DIR="${SCALE_ROOT}/samples"
DEVICE_ID=0
SEED=42
NUM_STEPS=50
SOLVER="midpoint"
FPS=8
# -----------------------------------------------------------------------------

validate_model_config
require_file "${REFERENCE_PATH}" "reference image or video"
require_file "${AUTOENCODER_CKPT}" "scale geometry autoencoder checkpoint"
require_file "${I0_CKPT}" "scale I0 decoder checkpoint"
require_file "${GENERATOR_CKPT}" "scale diffusion checkpoint"

ASCEND_RT_VISIBLE_DEVICES="${DEVICE_ID}" python3 \
  "${PROJECT}/sample_compact_i0.py" \
  --i0_path "${REFERENCE_PATH}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --diffusion_ckpt "${GENERATOR_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --solver "${SOLVER}" \
  --fps "${FPS}" \
  --seed "${SEED}"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/spatialvid.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-wan_compact_i2v14b480p_v1}"
AUTOENCODER_CKPT="${SCALE_GEOMETRY_AE_CKPT}"
I0_CKPT="${SCALE_I0_DECODER_CKPT}"
GENERATOR_CKPT="${SCALE_ROOT}/${EXPERIMENT_NAME}/checkpoint_latest.pt"
GENERATOR_CKPT_URL="${SCALE_REMOTE_ROOT}/${EXPERIMENT_NAME}/checkpoint_latest.pt"
GENERATOR_MIRROR_CKPT_URL="${SCALE_MIRROR_ROOT}/${EXPERIMENT_NAME}/checkpoint_latest.pt"
OUT_DIR="${SCALE_ROOT}/samples_${EXPERIMENT_NAME}"
REMOTE_OUT_DIR="${SCALE_REMOTE_ROOT}/samples_${EXPERIMENT_NAME}"
MIRROR_OUT_DIR="${SCALE_MIRROR_ROOT}/samples_${EXPERIMENT_NAME}"
SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-50}"
SOLVER="${SOLVER:-midpoint}"
FPS="${FPS:-8}"
WAN_CKPT_DIR="${WAN_CKPT_DIR}"

configure_modelarts_distributed
if [[ "${NODE_RANK}" -ne 0 ]]; then
  exit 0
fi

if [[ ! -d "${WAN_CKPT_DIR}" ]]; then
  echo "Wan checkpoint directory not found: ${WAN_CKPT_DIR}" >&2
  exit 1
fi

validate_model_config
ensure_spatialvid_splits
ensure_local_checkpoint \
  "${AUTOENCODER_CKPT}" "${SCALE_GEOMETRY_AE_CKPT_URL}" \
  "scale geometry autoencoder checkpoint" \
  "${SCALE_GEOMETRY_AE_MIRROR_CKPT_URL}"
ensure_local_checkpoint \
  "${I0_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
  "scale I0 decoder checkpoint" \
  "${SCALE_I0_DECODER_MIRROR_CKPT_URL}"
ensure_local_checkpoint \
  "${GENERATOR_CKPT}" "${GENERATOR_CKPT_URL}" \
  "Wan compact diffusion checkpoint" \
  "${GENERATOR_MIRROR_CKPT_URL}"

"${PYTHON_BIN}" "${PROJECT}/sample_compact_i0.py" \
  --csv "${SPATIALVID_EVAL_CSV}" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --encoder_ckpt "${STREAMVGGT_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_CKPT}" \
  --diffusion_ckpt "${GENERATOR_CKPT}" \
  --wan_ckpt_dir "${WAN_CKPT_DIR}" \
  --out_dir "${OUT_DIR}" \
  --num_steps "${NUM_STEPS}" \
  --solver "${SOLVER}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --dtype fp16

"${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
  "${OUT_DIR}" "${REMOTE_OUT_DIR}" --directory
if [[ -n "${MIRROR_OUT_DIR}" && "${MIRROR_OUT_DIR%/}" != "${REMOTE_OUT_DIR%/}" ]]; then
  "${PYTHON_BIN}" "${PROJECT}/scripts/moxing_transfer.py" \
    "${OUT_DIR}" "${MIRROR_OUT_DIR}" --directory
fi

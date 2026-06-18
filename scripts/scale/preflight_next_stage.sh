#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/../spatialvid_config.sh"
source "${SCRIPT_DIR}/../lib/modelarts.sh"

# Usage:
#   bash scripts/scale/preflight_next_stage.sh after_i0
#   bash scripts/scale/preflight_next_stage.sh before_merge
#   bash scripts/scale/preflight_next_stage.sh before_diffusion
#   bash scripts/scale/preflight_next_stage.sh before_sample
STAGE="${1:-after_i0}"
CACHE_NUM_PARTITIONS=1

configure_modelarts_distributed

case "${STAGE}" in
  after_i0|before_diffusion|before_sample)
    ensure_local_checkpoint \
      "${SCALE_GEOMETRY_AE_CKPT}" "${SCALE_GEOMETRY_AE_CKPT_URL}" \
      "scale geometry autoencoder checkpoint"
    ensure_local_checkpoint \
      "${SCALE_I0_DECODER_CKPT}" "${SCALE_I0_DECODER_CKPT_URL}" \
      "scale I0 decoder checkpoint"
    ;;
  before_merge)
    ;;
  *)
    echo "Unknown stage: ${STAGE}" >&2
    exit 2
    ;;
esac

if [[ "${STAGE}" == "before_sample" ]]; then
  ensure_local_checkpoint \
    "${SCALE_DIFFUSION_CKPT}" "${SCALE_DIFFUSION_CKPT_URL}" \
    "scale Compact DiT checkpoint"
fi

ARGS=(
  --stage "${STAGE}"
  --latent_dim "${SCALE_LATENT_DIM}"
  --latent_grid "${SCALE_LATENT_GRID}"
  --seq_len 8
  --train_cache_dir "${SCALE_TRAIN_CACHE_DIR}"
  --eval_cache_dir "${SCALE_EVAL_CACHE_DIR}"
  --cache_partitions "${CACHE_NUM_PARTITIONS}"
)

if [[ "${STAGE}" != "before_merge" ]]; then
  ARGS+=(
    --encoder_ckpt "${STREAMVGGT_CKPT}"
    --autoencoder_ckpt "${SCALE_GEOMETRY_AE_CKPT}"
    --i0_decoder_ckpt "${SCALE_I0_DECODER_CKPT}"
  )
fi
if [[ "${STAGE}" == "before_sample" ]]; then
  ARGS+=(--diffusion_ckpt "${SCALE_DIFFUSION_CKPT}")
fi

"${PYTHON_BIN}" "${PROJECT}/validate_scale_artifacts.py" "${ARGS[@]}"

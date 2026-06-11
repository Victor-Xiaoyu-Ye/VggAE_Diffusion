#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
I0_PATH=${I0_PATH:?Set I0_PATH to a reference image or video}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
I0_DECODER_CKPT=${I0_DECODER_CKPT:?Set I0_DECODER_CKPT}
DIFFUSION_CKPT=${DIFFUSION_CKPT:?Set DIFFUSION_CKPT}
OUT_DIR=${OUT_DIR:-${PROJECT}/outputs/compact_i0}
GPU_ID=${GPU_ID:-0}
SEED=${SEED:-42}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${PROJECT}/sample_compact_i0.py" \
  --i0_path "${I0_PATH}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --i0_decoder_ckpt "${I0_DECODER_CKPT}" \
  --diffusion_ckpt "${DIFFUSION_CKPT}" \
  --out_dir "${OUT_DIR}" \
  --num_steps 50 --solver midpoint --seed "${SEED}"

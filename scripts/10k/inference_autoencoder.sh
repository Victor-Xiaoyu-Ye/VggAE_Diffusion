#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to an evaluation metadata CSV}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/outputs/autoencoder_inference}
NUM_VIDEOS=${NUM_VIDEOS:-20}
GPU_ID=${GPU_ID:-0}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${PROJECT}/inference_autoencoder.py" \
  --checkpoint "${AUTOENCODER_CKPT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_videos "${NUM_VIDEOS}" \
  --seq_len 8 --target_size 518 \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --compute_psnr

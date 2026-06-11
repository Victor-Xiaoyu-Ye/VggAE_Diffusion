#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to representative evaluation metadata}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
OUTPUT=${OUTPUT:-${PROJECT}/outputs/diagnostics/latent_contract.json}
NUM_VIDEOS=${NUM_VIDEOS:-32}
GPU_ID=${GPU_ID:-0}

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 \
  "${PROJECT}/diagnose_latent_contract.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output "${OUTPUT}" \
  --num_videos "${NUM_VIDEOS}" \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 --num_workers 2

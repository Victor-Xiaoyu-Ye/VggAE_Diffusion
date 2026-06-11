#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to metadata containing the fixed validation clip}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/validation/i0_decoder_overfit}
GPU_ID=${GPU_ID:-0}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${CSV}" \
  --eval_video_root "${VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos 1 --disable_temporal_jitter \
  --latent_dim 512 --latent_grid 18 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs 500 --batch_size 1 --accum_steps 1 \
  --lr 3e-4 --wd 0 --warmup_steps 20 \
  --max_grad_norm 1.0 --lambda_lpips 0 \
  --dtype bf16 --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers 0 \
  --log_every 10 --save_every 25 --eval_every 25 \
  "${EXTRA_ARGS[@]}"

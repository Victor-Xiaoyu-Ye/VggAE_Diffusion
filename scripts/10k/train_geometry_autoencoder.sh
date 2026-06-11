#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to the training metadata CSV}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
EVAL_CSV=${EVAL_CSV:?Set EVAL_CSV to a held-out metadata CSV}
EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT:-${VIDEO_ROOT}}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/10k/geometry_autoencoder}
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29510}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
  --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_autoencoder.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${EVAL_CSV}" \
  --eval_video_root "${EVAL_VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --latent_noise_std 0.05 --latent_noise_warmup 1000 \
  --lambda_l1 1.0 --lambda_lpips 1.0 \
  --lambda_grad 0.05 --lambda_temporal 0.05 --lambda_latent_reg 0.01 \
  --batch_size 2 --accum_steps 15 \
  --epochs 120 --lr 1e-4 --wd 1e-2 \
  --warmup_steps 500 --ema_decay 0.999 \
  --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers 10 --dtype bf16 \
  --log_every 100 --eval_every 5 --save_every 5 \
  "${EXTRA_ARGS[@]}"

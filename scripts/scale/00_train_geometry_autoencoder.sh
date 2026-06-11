#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to the training metadata CSV}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the SpatialVID video root}
EVAL_CSV=${EVAL_CSV:?Set EVAL_CSV to a held-out metadata CSV}
EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT:-${VIDEO_ROOT}}
DEPTH_ROOT=${DEPTH_ROOT:?Set DEPTH_ROOT to aligned depth zip files}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29600}
REPRESENTATION_MAX_VIDEOS=${REPRESENTATION_MAX_VIDEOS:-1000000}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/scale/geometry_autoencoder}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_autoencoder.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${EVAL_CSV}" \
  --eval_video_root "${EVAL_VIDEO_ROOT}" \
  --depth_root "${DEPTH_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_videos "${REPRESENTATION_MAX_VIDEOS}" \
  --latent_dim 512 --latent_grid 18 \
  --levels 4 11 17 23 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --output_depth --lambda_depth 0.2 \
  --latent_noise_std 0.05 --latent_noise_warmup 5000 \
  --lambda_l1 1.0 --lambda_lpips 1.0 \
  --lambda_grad 0.05 --lambda_temporal 0.1 --lambda_latent_reg 0.01 \
  --batch_size 1 --accum_steps 8 \
  --epochs 3 --lr 1e-4 --wd 1e-2 \
  --warmup_steps 5000 --ema_decay 0.9999 \
  --seq_len 8 --target_size 518 \
  --max_frame_span 32 \
  --num_workers 8 --dtype bf16 \
  --log_every 100 --eval_every 1 --save_every 1 \
  "${EXTRA_ARGS[@]}"

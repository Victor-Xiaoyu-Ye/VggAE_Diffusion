#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CSV=${CSV:?Set CSV to the training metadata CSV}
VIDEO_ROOT=${VIDEO_ROOT:?Set VIDEO_ROOT to the video root}
EVAL_CSV=${EVAL_CSV:?Set EVAL_CSV to a held-out metadata CSV}
EVAL_VIDEO_ROOT=${EVAL_VIDEO_ROOT:-${VIDEO_ROOT}}
ENCODER_CKPT=${ENCODER_CKPT:?Set ENCODER_CKPT to StreamVGGT weights}
AUTOENCODER_CKPT=${AUTOENCODER_CKPT:?Set AUTOENCODER_CKPT}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/10k/i0_decoder}
NUM_GPUS=${NUM_GPUS:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MASTER_PORT=${MASTER_PORT:-29530}
RESUME=${RESUME:-}
EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
  --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_i0_autoencoder.py" \
  --csv "${CSV}" \
  --video_root "${VIDEO_ROOT}" \
  --eval_csv "${EVAL_CSV}" \
  --eval_video_root "${EVAL_VIDEO_ROOT}" \
  --encoder_ckpt "${ENCODER_CKPT}" \
  --autoencoder_ckpt "${AUTOENCODER_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 \
  --decoder_base_dim 384 --decoder_num_resblocks 2 \
  --epochs 50 --batch_size 1 --accum_steps 4 \
  --lr 1e-4 --wd 1e-2 --warmup_steps 500 \
  --max_grad_norm 1.0 --lambda_lpips 1.0 \
  --dtype bf16 --seq_len 8 --target_size 518 --max_frame_span 32 \
  --num_workers 2 \
  --log_every 50 --save_every 5 --eval_every 5 \
  "${EXTRA_ARGS[@]}"

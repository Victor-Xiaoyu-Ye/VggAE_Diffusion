#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CACHE_DIR=${CACHE_DIR:?Set CACHE_DIR to the merged latent cache}

NUM_GPUS=${NUM_GPUS:-8}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29604}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT}/ckpts/scale/compact_dit}
MAX_STEPS=${MAX_STEPS:-300000}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:?Set EVAL_CACHE_DIR to a held-out latent cache}
EVAL_I0_PATH=${EVAL_I0_PATH:-}
I0_DECODER_CKPT=${I0_DECODER_CKPT:-}
RESUME=${RESUME:-}
EXTRA_ARGS=()
EXTRA_ARGS+=(
  --eval_manifest "${EVAL_CACHE_DIR}/manifest.txt"
  --eval_stats "${EVAL_CACHE_DIR}/stats.pt"
)
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
fi
if [[ -n "${EVAL_I0_PATH}" || -n "${I0_DECODER_CKPT}" ]]; then
  : "${EVAL_I0_PATH:?Set both EVAL_I0_PATH and I0_DECODER_CKPT for RGB previews}"
  : "${I0_DECODER_CKPT:?Set both EVAL_I0_PATH and I0_DECODER_CKPT for RGB previews}"
  EXTRA_ARGS+=(
    --eval_i0_path "${EVAL_I0_PATH}"
    --i0_decoder_ckpt "${I0_DECODER_CKPT}"
  )
fi

torchrun --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT}/train_cached_compact_diffusion.py" \
  --manifest "${CACHE_DIR}/manifest.txt" \
  --stats "${CACHE_DIR}/stats.pt" \
  --output_dir "${OUTPUT_DIR}" \
  --latent_dim 512 --latent_grid 18 --seq_len 7 \
  --model_dim 768 --spatial_depth 8 --temporal_depth 4 --num_heads 12 \
  --time_scale 1000 \
  --batch_size 2 --accum_steps 8 \
  --max_steps "${MAX_STEPS}" \
  --lr 1e-4 --wd 1e-2 --warmup_steps 5000 \
  --ema_decay 0.9999 --max_grad_norm 1.0 \
  --num_workers 4 --shuffle_buffer 512 \
  --log_every 50 --save_every 10000 --eval_every 10000 \
  --sample_steps 20 --dtype bf16 \
  "${EXTRA_ARGS[@]}"
